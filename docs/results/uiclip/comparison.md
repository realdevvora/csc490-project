Subset: val (data/processed/splits_placeholder_85_15.json; 73 pages, 1,108 pairs)

| model | n | AUROC reg/benign | benign FPR@95R | threshold | precision | recall | F1 | benign FPR | no-change FPR | type macro-F1 | pointing | mean IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_cal_10_epochs_s0/best_auroc | 1108 | 0.991 | 0.021 | 0.182 | 0.995 | 0.952 | 0.973 | 0.021 | 0.014 | 0.377 | 0.425 | 0.204 |
| clip_cal_10_epochs_s1/best_auroc | 1108 | 0.991 | 0.007 | 0.408 | 0.989 | 0.975 | 0.982 | 0.055 | 0.027 | 0.276 | 0.354 | 0.193 |
| clip_cal_10_epochs_s2/best_auroc | 1108 | 0.995 | 0.000 | 0.177 | 0.990 | 0.973 | 0.981 | 0.041 | 0.041 | 0.369 | 0.431 | 0.220 |
| uiclip_cal_10_epochs_s0/best_auroc | 1108 | 0.994 | 0.000 | 0.179 | 0.995 | 0.969 | 0.982 | 0.014 | 0.027 | 0.382 | 0.427 | 0.207 |
| uiclip_cal_10_epochs_s1/best_auroc | 1108 | 0.991 | 0.014 | 0.326 | 0.991 | 0.972 | 0.981 | 0.041 | 0.027 | 0.327 | 0.438 | 0.201 |
| uiclip_cal_10_epochs_s2/best_auroc | 1108 | 0.994 | 0.014 | 0.116 | 0.985 | 0.984 | 0.985 | 0.068 | 0.041 | 0.384 | 0.450 | 0.204 |
| uiclip_cal_ft4_10_epochs_s0/best_auroc | 1108 | 0.999 | 0.007 | 0.372 | 0.990 | 0.998 | 0.994 | 0.041 | 0.041 | 0.450 | 0.475 | 0.179 |
| uiclip_cal_ft4_10_epochs_s1/best_auroc | 1108 | 0.999 | 0.007 | 0.267 | 0.993 | 0.999 | 0.996 | 0.021 | 0.041 | 0.443 | 0.485 | 0.188 |
| uiclip_cal_ft4_10_epochs_s2/best_auroc | 1108 | 0.998 | 0.014 | 0.268 | 0.990 | 0.999 | 0.994 | 0.041 | 0.041 | 0.448 | 0.458 | 0.190 |
| uiclip_cal_nobenign_10_epochs_s0/best_auroc | 1108 | 0.557 | 1.000 | 0.152 | 0.855 | 0.992 | 0.919 | 1.000 | 0.041 | 0.371 | 0.477 | 0.180 |
| uiclip_cal_nobenign_10_epochs_s1/best_auroc | 1108 | 0.661 | 1.000 | 0.158 | 0.855 | 0.990 | 0.918 | 1.000 | 0.041 | 0.395 | 0.468 | 0.187 |
| uiclip_cal_nobenign_10_epochs_s2/best_auroc | 1108 | 0.649 | 0.979 | 0.148 | 0.856 | 0.993 | 0.919 | 1.000 | 0.041 | 0.383 | 0.444 | 0.185 |
| uiclip_nocal_10_epochs_s0/best_auroc | 1108 | 0.994 | 0.000 | 0.203 | 0.984 | 0.981 | 0.983 | 0.075 | 0.041 | 0.329 | 0.466 | 0.205 |
| uiclip_nocal_10_epochs_s1/best_auroc | 1108 | 0.995 | 0.007 | 0.318 | 0.991 | 0.976 | 0.984 | 0.014 | 0.082 | 0.337 | 0.474 | 0.183 |
| uiclip_nocal_10_epochs_s2/best_auroc | 1108 | 0.994 | 0.000 | 0.871 | 0.997 | 0.971 | 0.983 | 0.000 | 0.041 | 0.390 | 0.465 | 0.229 |

- clip_cal_10_epochs_s0/best_auroc: threshold max F1 on the scored predictions
- clip_cal_10_epochs_s1/best_auroc: threshold max F1 on the scored predictions
- clip_cal_10_epochs_s2/best_auroc: threshold max F1 on the scored predictions
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
