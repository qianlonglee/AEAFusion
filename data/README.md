# AEAFusion data inventory

The manifests identify the files in the local AEAFusion data collection. To obtain the source data and check access conditions, visit [EMS](https://github.com/XunpengYi/EMS), [LLVIP](https://bupt-ai-cz.github.io/LLVIP/), [MFNet](https://www.mi.t.u-tokyo.ac.jp/projects/mil_multispectral/index.html) and [RoadScene](https://github.com/hanna-xu/RoadScene). EMS provides paired data and image degradation tools. Check the manifest hashes when matching a particular processed image to a local copy.

| Directory under the dataset root | Paired scenes | Files | Contents | Manifest |
| --- | ---: | ---: | --- | --- |
| `fusion/train` | 1,223 | 4,892 | Visible and infrared inputs and references | [fusion_train.tsv](manifests/fusion_train.tsv) |
| `fusion/test/LLVIP` | 200 | 600 | Two inputs and one pedestrian label file per scene | [llvip_test.tsv](manifests/llvip_test.tsv) |
| `fusion/test/MFNet` | 387 | 774 | Visible and infrared inputs | [mfnet_test.tsv](manifests/mfnet_test.tsv) |
| `exposure/train` | 169 | 676 | Visible and infrared inputs and references | [exposure_train.tsv](manifests/exposure_train.tsv) |
| `exposure/test` | 58 | 116 | Visible and infrared inputs | [exposure_test.tsv](manifests/exposure_test.tsv) |

The five manifests cover 7,058 files and 1,972,568,327 bytes. Each UTF-8 tab-separated row gives the path relative to the dataset root, the file size in bytes and the SHA-256 digest of the original file. Files are sorted by relative path. Re-encoding an image produces a different digest.

## Directory overlap

All 200 LLVIP and 387 MFNet test pairs also occur in the two input directories under `fusion/train`, with matching filenames and SHA-256 digests in both modalities. All 58 exposure test pairs likewise occur under `exposure/train`. The infrared input and reference in `exposure/train` are byte-identical for all 169 scenes. The directory names describe the stored collection. To construct a held-out evaluation, exclude each intended test pair from the training list before sampling.
