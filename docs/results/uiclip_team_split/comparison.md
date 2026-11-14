Subset: val (data/splits/splits_85_15.json)

| model | n | AUROC reg/benign | benign FPR@95R | threshold | precision | recall | F1 | benign FPR | no-change FPR | type macro-F1 | pointing | mean IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_cal_10_epochs_s0/best_auroc | 1033 | 0.994 | 0.022 | 0.083 | 0.988 | 0.983 | 0.985 | 0.072 | 0.000 | 0.393 | 0.460 | 0.231 |
| clip_cal_10_epochs_s1/best_auroc | 1033 | 0.984 | 0.094 | 0.178 | 0.976 | 0.973 | 0.975 | 0.145 | 0.000 | 0.416 | 0.446 | 0.224 |
| clip_cal_10_epochs_s2/best_auroc | 1033 | 0.987 | 0.051 | 0.645 | 0.985 | 0.962 | 0.974 | 0.087 | 0.000 | 0.387 | 0.436 | 0.225 |
| clip_cal_ft4_10_epochs_s0/best_auroc | 1033 | 1.000 | 0.000 | 0.310 | 0.996 | 1.000 | 0.998 | 0.022 | 0.000 | 0.397 | 0.471 | 0.198 |
| clip_cal_ft4_10_epochs_s1/best_auroc | 1033 | 1.000 | 0.000 | 0.276 | 1.000 | 0.998 | 0.999 | 0.000 | 0.000 | 0.486 | 0.492 | 0.217 |
| clip_cal_ft4_10_epochs_s2/best_auroc | 1033 | 1.000 | 0.000 | 0.082 | 0.999 | 0.999 | 0.999 | 0.007 | 0.000 | 0.458 | 0.527 | 0.223 |
| siglip_cal_10_epochs_s0/best_auroc | 1033 | 0.993 | 0.029 | 0.312 | 0.995 | 0.966 | 0.980 | 0.029 | 0.000 | 0.385 | 0.556 | 0.208 |
| siglip_cal_10_epochs_s1/best_auroc | 1033 | 0.998 | 0.014 | 0.190 | 0.994 | 0.992 | 0.993 | 0.036 | 0.000 | 0.420 | 0.557 | 0.220 |
| siglip_cal_10_epochs_s2/best_auroc | 1033 | 0.998 | 0.000 | 0.178 | 1.000 | 0.953 | 0.976 | 0.000 | 0.000 | 0.393 | 0.559 | 0.202 |
| siglip_cal_ft4_10_epochs_s0/best_auroc | 1033 | 1.000 | 0.000 | 0.296 | 0.998 | 0.996 | 0.997 | 0.014 | 0.000 | 0.473 | 0.610 | 0.205 |
| uiclip_cal_10_epochs_s0/best_auroc | 1033 | 0.991 | 0.014 | 0.452 | 0.994 | 0.970 | 0.982 | 0.036 | 0.000 | 0.385 | 0.454 | 0.233 |
| uiclip_cal_10_epochs_s1/best_auroc | 1033 | 0.988 | 0.014 | 0.279 | 0.993 | 0.961 | 0.977 | 0.043 | 0.000 | 0.334 | 0.448 | 0.233 |
| uiclip_cal_10_epochs_s2/best_auroc | 1033 | 0.984 | 0.014 | 0.252 | 0.990 | 0.969 | 0.979 | 0.058 | 0.000 | 0.392 | 0.464 | 0.225 |
| uiclip_cal_ft4_10_epochs_s0/best_auroc | 1033 | 1.000 | 0.000 | 0.191 | 0.999 | 1.000 | 0.999 | 0.007 | 0.000 | 0.494 | 0.482 | 0.227 |
| uiclip_cal_ft4_10_epochs_s1/best_auroc | 1033 | 1.000 | 0.000 | 0.440 | 0.999 | 0.996 | 0.998 | 0.007 | 0.000 | 0.352 | 0.495 | 0.239 |
| uiclip_cal_ft4_10_epochs_s2/best_auroc | 1033 | 1.000 | 0.000 | 0.121 | 1.000 | 0.994 | 0.997 | 0.000 | 0.000 | 0.479 | 0.510 | 0.232 |
| uiclip_cal_nobenign_10_epochs_s0/best_auroc | 1033 | 0.488 | 1.000 | 0.330 | 0.857 | 1.000 | 0.923 | 1.000 | 0.000 | 0.401 | 0.487 | 0.190 |
| uiclip_cal_nobenign_10_epochs_s1/best_auroc | 1033 | 0.487 | 1.000 | 0.152 | 0.857 | 1.000 | 0.923 | 1.000 | 0.000 | 0.373 | 0.484 | 0.192 |
| uiclip_cal_nobenign_10_epochs_s2/best_auroc | 1033 | 0.484 | 1.000 | 0.147 | 0.857 | 1.000 | 0.923 | 1.000 | 0.000 | 0.397 | 0.450 | 0.203 |
| uiclip_nocal_10_epochs_s0/best_auroc | 1033 | 0.990 | 0.000 | 0.570 | 0.996 | 0.967 | 0.982 | 0.022 | 0.000 | 0.336 | 0.485 | 0.217 |
| uiclip_nocal_10_epochs_s1/best_auroc | 1033 | 0.992 | 0.007 | 0.176 | 0.985 | 0.977 | 0.981 | 0.087 | 0.000 | 0.384 | 0.462 | 0.235 |
| uiclip_nocal_10_epochs_s2/best_auroc | 1033 | 0.990 | 0.014 | 0.522 | 0.989 | 0.972 | 0.980 | 0.065 | 0.000 | 0.418 | 0.483 | 0.212 |

- clip_cal_10_epochs_s0/best_auroc: threshold max F1 on the scored predictions
- clip_cal_10_epochs_s1/best_auroc: threshold max F1 on the scored predictions
- clip_cal_10_epochs_s2/best_auroc: threshold max F1 on the scored predictions
- clip_cal_ft4_10_epochs_s0/best_auroc: threshold max F1 on the scored predictions
- clip_cal_ft4_10_epochs_s1/best_auroc: threshold max F1 on the scored predictions
- clip_cal_ft4_10_epochs_s2/best_auroc: threshold max F1 on the scored predictions
- siglip_cal_10_epochs_s0/best_auroc: threshold max F1 on the scored predictions
- siglip_cal_10_epochs_s1/best_auroc: threshold max F1 on the scored predictions
- siglip_cal_10_epochs_s2/best_auroc: threshold max F1 on the scored predictions
- siglip_cal_ft4_10_epochs_s0/best_auroc: threshold max F1 on the scored predictions
- uiclip_cal_10_epochs_s0/best_auroc: threshold max F1 on the scored predictions
- uiclip_cal_10_epochs_s1/best_auroc: threshold max F1 on the scored predictions
- uiclip_cal_10_epochs_s2/best_auroc: threshold max F1 on the scored predictions
- uiclip_cal_ft4_10_epochs_s0/best_auroc: threshold max F1 on the scored predictions
- uiclip_cal_ft4_10_epochs_s1/best_auroc: threshold max F1 on the scored predictions
- uiclip_cal_ft4_10_epochs_s2/best_auroc: threshold max F1 on the scored predictions
- uiclip_cal_nobenign_10_epochs_s0/best_auroc: threshold max F1 on the scored predictions
- uiclip_cal_nobenign_10_epochs_s1/best_auroc: threshold max F1 on the scored predictions
- uiclip_cal_nobenign_10_epochs_s2/best_auroc: threshold max F1 on the scored predictions
- uiclip_nocal_10_epochs_s0/best_auroc: threshold max F1 on the scored predictions
- uiclip_nocal_10_epochs_s1/best_auroc: threshold max F1 on the scored predictions
- uiclip_nocal_10_epochs_s2/best_auroc: threshold max F1 on the scored predictions
