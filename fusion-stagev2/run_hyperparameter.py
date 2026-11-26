"""AEAFusion: 13 one-factor-at-a-time Stage-2 runs, 100 epochs, seed 42.

Put this file beside train_stage2_cara_full.py. Uses a frozen source snapshot,
preserves the original training math, and evaluates epoch 100 for every run.
Training, inference and metrics run in separate processes to release GPU memory.
"""
import argparse
import ast
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/msia/Leeql/LYTFusion/Code_CARA/fusion-stagev2')
DATA = Path('/home/msia/Leeql/EMS_full_dataset/EMS_dataset_train/vis_Low-Light')
SOURCE_FILES = ['train_stage2_cara_full.py', 'dataloader_fusion.py', 'model.py',
                'fusion_net_cara_full.py', 'losses_fusion_cara_full.py', 'utils.py',
                'eval_denoif_metrics.py']
METRICS = ['SSIM', 'PSNR', 'FSIM', 'CC', 'BRISQUE']


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(str(tmp), str(path))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def design():
    base = dict(LAMBDA_CARA=0.10, REJECT_STRENGTH=0.60, RES_SCALE=0.45)
    rows = [dict(id='baseline', parameter='baseline', value=None, **base)]
    for key, label, values in [
        ('LAMBDA_CARA', 'lambda_aea', [0.025, 0.05, 0.10, 0.20, 0.40]),
        ('REJECT_STRENGTH', 'beta', [0.20, 0.40, 0.60, 0.80, 1.00]),
        ('RES_SCALE', 'rho', [0.15, 0.30, 0.45, 0.60, 0.75]),
    ]:
        for value in values:
            if value == base[key]:
                continue
            row = dict(base)
            row[key] = value
            rows.append(dict(id=label + '_' + str(value).replace('.', 'p'),
                             parameter=label, value=value, **row))
    return rows


def prepare(args):
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    config_file = out / 'sweep_config.json'
    if not config_file.exists():
        source = out / 'source_snapshot'
        source.mkdir(exist_ok=True)
        for name in SOURCE_FILES:
            shutil.copy2(str(args.source_root / name), str(source / name))
        constants = {}
        for node in ast.parse((source / SOURCE_FILES[0]).read_text(encoding='utf-8')).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.isupper():
                    try:
                        constants[name] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        constants.update(EPOCHS=100, SEED=42, SAVE_MODEL_EVERY=100, SAVE_BEST_BY_LOSS=False,
                         VIS_LOW_DIR=str(DATA / 'Visible_LQ'),
                         IR_NOISY_DIR=str(DATA / 'Infrared_gt_noisy1'),
                         RESTORER_CKPT='/home/msia/Leeql/LYTFusion/V4.3.1/best_model.pth')
        config = dict(training=constants, runs=design(),
                      test_vis=str(DATA / 'LLVIP200/VIS_LQ'),
                      test_ir=str(DATA / 'LLVIP200/Infrared_gt_noisy1'),
                      clean_vis=str(DATA / 'LLVIP200/VIS_HQ'),
                      clean_ir=str(DATA / 'LLVIP200/Infrared_gt'),
                      checkpoint_rule='fixed_epoch_100',
                      repetitions='one Stage-2 training seed; not a repeated-seed study',
                      padding='right/bottom replication padding to a multiple of 32; crop output back',
                      visual_samples='original forward/sample schedule retained; image-grid writes disabled',
                      source_sha256={name: sha256(source / name) for name in SOURCE_FILES},
                      restorer_sha256=sha256(constants['RESTORER_CKPT']))
        write_json(config_file, config)
        shutil.copy2(__file__, str(out / 'runner_snapshot.py'))
    config = json.loads(config_file.read_text(encoding='utf-8'))
    if 'original_restorer_ckpt' not in config:
        original = Path(config['training']['RESTORER_CKPT'])
        frozen = out / 'source_snapshot' / 'restorer_fixed.pth'
        shutil.copy2(str(original), str(frozen))
        if sha256(frozen) != config['restorer_sha256']:
            raise RuntimeError('Restorer checkpoint changed since experiment preparation.')
        config['original_restorer_ckpt'] = str(original)
        config['training']['RESTORER_CKPT'] = str(frozen)
        write_json(config_file, config)
    for name, expected in config['source_sha256'].items():
        if sha256(out / 'source_snapshot' / name) != expected:
            raise RuntimeError('Source snapshot was changed: ' + name)
    if sha256(config['training']['RESTORER_CKPT']) != config['restorer_sha256']:
        raise RuntimeError('Frozen restorer checkpoint was changed.')
    return config


def load_training(args, config, row):
    import train_stage2_cara_full as train
    settings = dict(config['training'])
    settings.update({k: row[k] for k in ['LAMBDA_CARA', 'REJECT_STRENGTH', 'RES_SCALE']})
    settings['OUT_DIR'] = str(args.output_root / row['id'])
    for key, value in settings.items():
        setattr(train, key, value)
    # Preserve the original extra forwards and data/RNG consumption, saving disk only.
    train.save_visual_samples = lambda *a, **kw: None
    return train, settings


def checkpoint(args, row):
    return args.output_root / row['id'] / 'checkpoints/fusion_cara_full_epoch_0100.pth'


def make_models(train, device):
    import torch
    restorer = train.LYT_DualBranch().to(device)
    state = torch.load(train.RESTORER_CKPT, map_location='cpu', weights_only=True)
    restorer.load_state_dict(state.get('state_dict', state), strict=True)
    restorer.eval().requires_grad_(False)
    fusion = train.FusionNetCARAFull(base_channels=train.BASE_CHANNELS, res_scale=train.RES_SCALE,
                                   output_base=train.OUTPUT_BASE, reject_strength=train.REJECT_STRENGTH).to(device)
    return restorer, fusion


def preflight(args, config):
    import torch
    import piq
    from dataloader_fusion import create_fusion_dataloader
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this sweep.')
    train, _ = load_training(args, config, config['runs'][0])
    loader = create_fusion_dataloader(train.VIS_LOW_DIR, train.IR_NOISY_DIR,
                                     crop_size=train.CROP_SIZE, batch_size=train.BATCH_SIZE,
                                     num_workers=0, training=True, strict_filename_match=True)
    if not len(loader):
        raise RuntimeError('Training loader has no full batch.')
    make_models(train, 'cpu')
    report = dict(torch=str(torch.__version__), piq=piq.__version__, gpu=torch.cuda.get_device_name(0),
                  train_pairs=len(loader.dataset), batches_per_epoch=len(loader),
                  batch_size=train.BATCH_SIZE, epochs=train.EPOCHS, configurations=len(config['runs']),
                  test_inputs_available=test_ready(config),
                  free_gib=shutil.disk_usage(args.output_root).free / 1024**3)
    write_json(args.output_root / 'preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


def smoke(args, config):
    import torch
    train, _ = load_training(args, config, config['runs'][0])
    train.set_seed(train.SEED)
    loader = train.create_fusion_dataloader(train.VIS_LOW_DIR, train.IR_NOISY_DIR,
                                           crop_size=train.CROP_SIZE, batch_size=train.BATCH_SIZE,
                                           num_workers=0, training=True)
    restorer, fusion = make_models(train, 'cuda')
    vi, ir = [x.cuda() for x in next(iter(loader))]
    with torch.no_grad():
        vh, ih, vc, ic = [x.clamp(0, 1) for x in restorer(vi, ir)]
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=train.LR, betas=(0.9, 0.999), weight_decay=1e-4)
    _, fu, maps = fusion(vh, ih, vc, ic, return_feats=True, return_maps=True)
    base, _, _ = train.fusion_loss_base(vi=vh, ir=ih, fu=fu, w_int=train.W_INT, w_grad=train.W_GRAD)
    feat = train.max_evidence_feature_loss(maps['r2'], maps['t2'], maps['f2'])
    aea, _ = train.cara_full_regularization(maps)
    low = train.low_frequency_consistency_loss(vh, ih, fu)
    loss = base + train.LAMBDA_FEAT * feat + train.LAMBDA_CARA * aea + train.LAMBDA_LOW * low
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite smoke-test loss.')
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError('Non-finite smoke-test gradients.')
    optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps(dict(smoke='passed', batch_size=vi.shape[0], loss=float(loss.detach()),
                          gradient_norm=float(grad_norm),
                          peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)), flush=True)


def train_worker(args, config, row):
    import torch
    train, settings = load_training(args, config, row)
    run_dir = args.output_root / row['id']
    run_dir.mkdir(exist_ok=True)
    write_json(run_dir / 'config.json', dict(settings=settings, experiment=row,
                                           source_sha256=config['source_sha256'],
                                           restorer_sha256=config['restorer_sha256']))
    train.main()
    state = torch.load(checkpoint(args, row), map_location='cpu', weights_only=True)
    if state['epoch'] != 100:
        raise RuntimeError('Expected epoch 100.')
    state['hyperparameters'] = settings
    state['restorer_sha256'] = config['restorer_sha256']
    path = checkpoint(args, row)
    torch.save(state, str(path) + '.tmp')
    os.replace(str(path) + '.tmp', str(path))
    write_json(run_dir / 'training_complete.json', dict(epoch=100, checkpoint=str(path),
                                                       sha256=sha256(path), finished_at=time.time()))


def test_ready(config):
    return all(Path(config[k]).is_dir() and any(Path(config[k]).iterdir()) for k in ['test_vis', 'test_ir'])


def pad_pair(vi, ir):
    import torch.nn.functional as F
    if vi.shape[-2:] != ir.shape[-2:]:
        raise ValueError('VIS/IR spatial dimensions differ.')
    h, w = vi.shape[-2:]
    pad = (0, (-w) % 32, 0, (-h) % 32)
    if pad[1] or pad[3]:
        vi, ir = F.pad(vi, pad, mode='replicate'), F.pad(ir, pad, mode='replicate')
    return vi, ir, (h, w)


def infer_worker(args, config, row):
    import torch
    from torchvision.utils import save_image
    train, _ = load_training(args, config, row)
    loader = train.create_fusion_dataloader(config['test_vis'], config['test_ir'], crop_size=None,
                                           batch_size=1, num_workers=2, training=False,
                                           strict_filename_match=True)
    restorer, fusion = make_models(train, 'cuda')
    state = torch.load(checkpoint(args, row), map_location='cpu', weights_only=True)
    for key in ['LAMBDA_CARA', 'REJECT_STRENGTH', 'RES_SCALE']:
        if state['hyperparameters'][key] != row[key]:
            raise RuntimeError('Training/inference hyperparameter mismatch: ' + key)
    fusion.load_state_dict(state['fusion'], strict=True)
    fusion.eval()
    out = args.output_root / row['id'] / 'results_LLVIP_100'
    out.mkdir(exist_ok=True)
    with torch.no_grad():
        for i, (vi, ir) in enumerate(loader):
            vi, ir, (h, w) = pad_pair(vi.cuda(), ir.cuda())
            vh, ih, vc, ic = [x.clamp(0, 1) for x in restorer(vi, ir)]
            _, fused, _ = fusion(vh, ih, vc, ic, return_maps=True)
            fused = fused[..., :h, :w]
            if not torch.isfinite(fused).all():
                raise RuntimeError('Non-finite fused output.')
            name = Path(loader.dataset.vis_list[i]).stem + '.png'
            save_image(fused, str(out / name))
            print('[{}/{}] {}'.format(i + 1, len(loader), name), flush=True)
    write_json(args.output_root / row['id'] / 'inference_complete.json',
               dict(images=len(loader.dataset), output_dir=str(out), test_vis=config['test_vis'],
                    test_ir=config['test_ir'], finished_at=time.time()))


def metrics_worker(args, config, row):
    import torch
    import eval_denoif_metrics as metrics
    if not metrics._HAS_PIQ:
        raise RuntimeError('PIQ is required; approximate/missing metrics are not allowed.')
    out = args.output_root / row['id']
    fused = metrics._stem_map(str(out / 'results_LLVIP_100'))
    visible = metrics._stem_map(config['clean_vis'])
    infrared = metrics._stem_map(config['clean_ir'])
    expected = set(metrics._stem_map(config['test_vis']))
    if set(fused) != expected or not expected.issubset(set(visible) & set(infrared)):
        raise RuntimeError('Missing or extra fused images / missing clean references.')
    vals = {k: [] for k in METRICS}
    rows = []
    funcs = dict(SSIM=metrics._ssim, PSNR=metrics._psnr, FSIM=metrics._fsim, CC=metrics._cc)
    with torch.no_grad():
        for i, stem in enumerate(sorted(expected), 1):
            rgb = metrics._load_rgb(fused[stem]).cuda()
            fy = metrics.rgb_to_y(rgb)
            vy = metrics.rgb_to_y(metrics._load_rgb(visible[stem]).cuda())
            iy = metrics._load_gray(infrared[stem]).cuda()
            scores = {k: 0.5 * fun(fy, vy) + 0.5 * fun(fy, iy) for k, fun in funcs.items()}
            scores['BRISQUE'] = metrics._brisque(rgb)
            if not all(__import__('math').isfinite(v) for v in scores.values()):
                raise RuntimeError('Non-finite metric for ' + stem)
            for key, value in scores.items():
                vals[key].append(value)
            rows.append(dict(image=stem, **scores))
            print('[{}/{}] {}'.format(i, len(expected), stem), flush=True)
    with open(out / 'metrics_per_image.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image'] + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: dict(zip(['mean', 'std_images_ddof0'], metrics._mean_std(vals[key]))) for key in METRICS}
    write_json(out / 'metrics_summary.json', dict(images=len(rows), metrics=summary,
                                                 aggregation='equal-weight clean VIS/IR metric scores; std across images',
                                                 finished_at=time.time()))
    print(json.dumps(summary, indent=2), flush=True)


def aggregate(args, config):
    rows = []
    for run in config['runs']:
        row = dict(run)
        path = args.output_root / run['id'] / 'metrics_summary.json'
        if path.exists():
            data = json.loads(path.read_text())
            row['images'] = data['images']
            for metric in METRICS:
                for stat, value in data['metrics'][metric].items():
                    row[metric + '_' + stat] = value
        rows.append(row)
    fields = ['id', 'parameter', 'value', 'LAMBDA_CARA', 'REJECT_STRENGTH', 'RES_SCALE', 'images']
    fields += [k + '_' + stat for k in METRICS for stat in ['mean', 'std_images_ddof0']]
    path = args.output_root / 'metrics_summary.csv'
    with open(str(path) + '.tmp', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(path) + '.tmp', str(path))


def launch_worker(args, phase, run):
    folder = args.output_root / run['id']
    folder.mkdir(exist_ok=True)
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--phase', phase,
           '--output-root', str(args.output_root), '--source-root', str(args.source_root), '--run-id', run['id']]
    print(time.strftime('%F %T'), phase, run['id'], flush=True)
    with open(folder / (phase + '.log'), 'a', buffering=1) as log:
        log.write('\n=== Started {} ===\n'.format(time.strftime('%F %T')))
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        write_json(args.output_root / 'status.json', dict(status=phase, run=run['id'], pid=child.pid,
                                                         supervisor_pid=os.getpid(), updated_at=time.time()))
        code = child.wait()
    if code:
        raise RuntimeError('{} failed for {} (exit {}); see {}.log'.format(phase, run['id'], code, phase))


def run_sweep(args, config):
    with open(args.output_root / 'sweep.lock', 'a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for run in config['runs']:
                if shutil.disk_usage(args.output_root).free < 2 * 1024**3:
                    raise RuntimeError('Less than 2 GiB free disk space; stopped without deleting any files.')
                folder = args.output_root / run['id']
                if not (folder / 'training_complete.json').exists():
                    launch_worker(args, 'train', run)
                config = json.loads((args.output_root / 'sweep_config.json').read_text())
                if test_ready(config):
                    if not (folder / 'inference_complete.json').exists():
                        launch_worker(args, 'infer', run)
                    if not (folder / 'metrics_summary.json').exists():
                        launch_worker(args, 'metrics', run)
                else:
                    print('Testing pending: input directories unavailable.', flush=True)
                aggregate(args, config)
            pending = [r['id'] for r in config['runs'] if not (args.output_root / r['id'] / 'metrics_summary.json').exists()]
            write_json(args.output_root / 'status.json', dict(status='awaiting_test_inputs' if pending else 'complete',
                                                             pending=pending, updated_at=time.time()))
        except BaseException as exc:
            write_json(args.output_root / 'status.json', dict(status='failed', error=str(exc), updated_at=time.time()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['plan', 'preflight', 'smoke', 'run', 'train', 'infer', 'metrics'], default='run')
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs_Hyperparameter')
    parser.add_argument('--run-id', default='baseline')
    args = parser.parse_args()
    if args.phase == 'plan':
        print(json.dumps(design(), indent=2))
        return
    config = prepare(args)
    sys.path.insert(0, str(args.output_root / 'source_snapshot'))
    if args.phase == 'preflight':
        preflight(args, config)
    elif args.phase == 'smoke':
        smoke(args, config)
    elif args.phase == 'run':
        run_sweep(args, config)
    else:
        row = next(r for r in config['runs'] if r['id'] == args.run_id)
        {'train': train_worker, 'infer': infer_worker, 'metrics': metrics_worker}[args.phase](args, config, row)


if __name__ == '__main__':
    main()
