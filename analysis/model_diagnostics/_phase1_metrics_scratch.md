# Phase-1 Variant Metrics Scratch

All metrics were recomputed directly from existing LOMO prediction NPZs. Per-fold finite validation occurs before aggregation. Headlines use median across folds; means are reported separately.

## Prediction inventory

| model | target | seed | available_folds |
|---|---|---|---|
| hpg_hier | EA | 42 | 9 |
| hpg_hier | IP | 42 | 9 |
| hpg_hier_junction | EA | 42 | 9 |
| hpg_hier_junction | IP | 42 | 9 |
| hpg_hier_junction1 | EA | 42 | 9 |
| hpg_hier_junction1 | IP | 42 | 9 |
| hpg_hier_octamer | EA | 42 | 9 |
| hpg_hier_octamer | IP | 42 | 9 |
| wdmpnn | EA | 42 | 9 |
| wdmpnn | IP | 42 | 9 |

Missing requested model-target-seed-fold cells: 180. No non-finite prediction arrays were found among loaded cells: 0.

## Seed-42 LOMO aggregate metrics

| model | target | group_mean_r2_median | group_mean_r2_mean | delta_r2_median | delta_r2_mean | ordering_median | ordering_mean | overall_r2_median | overall_r2_mean | overall_mae_median | overall_mae_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hpg_hier | EA | 0.92534 | 0.89976 | 0.79027 | 0.68558 | 0.79896 | 0.78124 | 0.92411 | 0.89252 | 0.09393 | 0.10365 |
| hpg_hier_junction | EA | 0.97099 | 0.94875 | 0.78790 | 0.61345 | 0.78804 | 0.76060 | 0.96887 | 0.93659 | 0.08649 | 0.07678 |
| hpg_hier_junction1 | EA | 0.96279 | 0.88348 | 0.78547 | 0.72181 | 0.80254 | 0.77146 | 0.96125 | 0.87195 | 0.05864 | 0.08525 |
| hpg_hier_octamer | EA | 0.98949 | 0.97317 | 0.78890 | 0.68365 | 0.81826 | 0.79481 | 0.98633 | 0.96565 | 0.04256 | 0.04922 |
| wdmpnn | EA | 0.96479 | 0.93802 | 0.58024 | 0.51145 | 0.75301 | 0.72220 | 0.95590 | 0.92765 | 0.06855 | 0.08498 |
| hpg_hier | IP | 0.96862 | 0.92113 | 0.81947 | 0.73395 | 0.82568 | 0.81795 | 0.96311 | 0.91709 | 0.05101 | 0.07242 |
| hpg_hier_junction | IP | 0.96175 | 0.89374 | 0.77508 | 0.76566 | 0.80938 | 0.81143 | 0.96060 | 0.89019 | 0.04954 | 0.07375 |
| hpg_hier_junction1 | IP | 0.95209 | 0.86756 | 0.76228 | 0.71941 | 0.81004 | 0.80312 | 0.94217 | 0.86178 | 0.05775 | 0.07695 |
| hpg_hier_octamer | IP | 0.96971 | 0.95195 | 0.86802 | 0.79819 | 0.82706 | 0.82039 | 0.96739 | 0.94686 | 0.05369 | 0.05075 |
| wdmpnn | IP | 0.96794 | 0.86440 | 0.46005 | 0.54903 | 0.75774 | 0.75828 | 0.95694 | 0.84665 | 0.06211 | 0.07765 |

## Junction depth: baseline vs n=1 vs n=2 — seed-42 per-fold metrics

| target | fold | model | group_mean_r2 | delta_r2 | ordering | overall_r2 | overall_mae | finite |
|---|---|---|---|---|---|---|---|---|
| EA | 0 | hpg_hier | 0.92534 | 0.73356 | 0.74454 | 0.92411 | 0.10681 | True |
| EA | 0 | hpg_hier_junction | 0.91531 | 0.77177 | 0.78804 | 0.91221 | 0.11850 | True |
| EA | 0 | hpg_hier_junction1 | 0.98889 | 0.78547 | 0.75318 | 0.98624 | 0.04645 | True |
| EA | 1 | hpg_hier | 0.57514 | 0.79027 | 0.81215 | 0.56850 | 0.21397 | True |
| EA | 1 | hpg_hier_junction | 0.92462 | 0.76452 | 0.72353 | 0.91801 | 0.08649 | True |
| EA | 1 | hpg_hier_junction1 | 0.96187 | 0.81346 | 0.80564 | 0.95671 | 0.05785 | True |
| EA | 2 | hpg_hier | 0.92157 | 0.84550 | 0.80564 | 0.92365 | 0.11819 | True |
| EA | 2 | hpg_hier_junction | 0.97848 | 0.83113 | 0.77533 | 0.97646 | 0.06545 | True |
| EA | 2 | hpg_hier_junction1 | 0.90371 | 0.67238 | 0.75122 | 0.90259 | 0.13719 | True |
| EA | 3 | hpg_hier | 0.97464 | 0.83719 | 0.79896 | 0.96855 | 0.05469 | True |
| EA | 3 | hpg_hier_junction | 0.98604 | 0.80348 | 0.86869 | 0.98004 | 0.04058 | True |
| EA | 3 | hpg_hier_junction1 | 0.98485 | 0.91174 | 0.86673 | 0.98188 | 0.04107 | True |
| EA | 4 | hpg_hier | 0.95060 | 0.34695 | 0.57804 | 0.93451 | 0.07334 | True |
| EA | 4 | hpg_hier_junction | 0.98012 | 0.56119 | 0.53552 | 0.97328 | 0.05222 | True |
| EA | 4 | hpg_hier_junction1 | 0.95515 | 0.58306 | 0.59840 | 0.94516 | 0.07432 | True |
| EA | 5 | hpg_hier | 0.96903 | 0.88197 | 0.84669 | 0.96482 | 0.05831 | True |
| EA | 5 | hpg_hier_junction | 0.97449 | 0.88365 | 0.84066 | 0.97176 | 0.04943 | True |
| EA | 5 | hpg_hier_junction1 | 0.97588 | 0.86871 | 0.80857 | 0.97192 | 0.04448 | True |
| EA | 6 | hpg_hier | 0.91655 | 0.24291 | 0.79635 | 0.88211 | 0.05805 | True |
| EA | 6 | hpg_hier_junction | 0.84377 | -0.68791 | 0.67286 | 0.76390 | 0.08893 | True |
| EA | 6 | hpg_hier_junction1 | 0.23157 | 0.29109 | 0.74014 | 0.15664 | 0.21154 | True |
| EA | 7 | hpg_hier | 0.90187 | 0.64959 | 0.79472 | 0.90497 | 0.15554 | True |
| EA | 7 | hpg_hier_junction | 0.96494 | 0.78790 | 0.81981 | 0.96476 | 0.09969 | True |
| EA | 7 | hpg_hier_junction1 | 0.98665 | 0.78343 | 0.80254 | 0.98515 | 0.05864 | True |
| EA | 8 | hpg_hier | 0.96315 | 0.84228 | 0.85402 | 0.96150 | 0.09393 | True |
| EA | 8 | hpg_hier_junction | 0.97099 | 0.80528 | 0.82095 | 0.96887 | 0.08970 | True |
| EA | 8 | hpg_hier_junction1 | 0.96279 | 0.78690 | 0.81672 | 0.96125 | 0.09576 | True |
| IP | 0 | hpg_hier | 0.92787 | 0.82822 | 0.82682 | 0.92315 | 0.06460 | True |
| IP | 0 | hpg_hier_junction | 0.98123 | 0.87665 | 0.83056 | 0.97650 | 0.03285 | True |
| IP | 0 | hpg_hier_junction1 | 0.87220 | 0.92258 | 0.86298 | 0.86813 | 0.08826 | True |
| IP | 1 | hpg_hier | 0.98116 | 0.73747 | 0.82568 | 0.97898 | 0.04499 | True |
| IP | 1 | hpg_hier_junction | 0.96175 | 0.77508 | 0.81753 | 0.96060 | 0.06280 | True |
| IP | 1 | hpg_hier_junction1 | 0.98131 | 0.78624 | 0.81004 | 0.97966 | 0.04407 | True |
| IP | 2 | hpg_hier | 0.76920 | 0.21126 | 0.75692 | 0.77432 | 0.19782 | True |
| IP | 2 | hpg_hier_junction | 0.87077 | 0.67182 | 0.76507 | 0.87333 | 0.15918 | True |
| IP | 2 | hpg_hier_junction1 | 0.91484 | 0.56432 | 0.77517 | 0.91550 | 0.12285 | True |
| IP | 3 | hpg_hier | 0.97463 | 0.81947 | 0.82193 | 0.96311 | 0.03347 | True |
| IP | 3 | hpg_hier_junction | 0.84529 | 0.73310 | 0.80938 | 0.83291 | 0.08116 | True |
| IP | 3 | hpg_hier_junction1 | 0.91361 | 0.85371 | 0.82405 | 0.90555 | 0.05970 | True |
| IP | 4 | hpg_hier | 0.94379 | 0.58074 | 0.76393 | 0.93211 | 0.04514 | True |
| IP | 4 | hpg_hier_junction | 0.92729 | 0.59595 | 0.80108 | 0.91982 | 0.04954 | True |
| IP | 4 | hpg_hier_junction1 | 0.95209 | 0.56024 | 0.73770 | 0.94217 | 0.04130 | True |
| IP | 5 | hpg_hier | 0.76969 | 0.89601 | 0.91675 | 0.76798 | 0.11849 | True |
| IP | 5 | hpg_hier_junction | 0.49409 | 0.76665 | 0.88025 | 0.49896 | 0.17561 | True |
| IP | 5 | hpg_hier_junction1 | 0.23951 | 0.56500 | 0.84327 | 0.23520 | 0.21233 | True |
| IP | 6 | hpg_hier | 0.97462 | 0.87979 | 0.84620 | 0.97164 | 0.05101 | True |
| IP | 6 | hpg_hier_junction | 0.99305 | 0.80191 | 0.80531 | 0.98786 | 0.03155 | True |
| IP | 6 | hpg_hier_junction1 | 0.99231 | 0.85862 | 0.80873 | 0.98846 | 0.02991 | True |
| IP | 7 | hpg_hier | 0.96862 | 0.77267 | 0.73053 | 0.96628 | 0.06580 | True |
| IP | 7 | hpg_hier_junction | 0.98008 | 0.83088 | 0.74845 | 0.97801 | 0.04767 | True |
| IP | 7 | hpg_hier_junction1 | 0.97384 | 0.76228 | 0.70398 | 0.97017 | 0.05775 | True |
| IP | 8 | hpg_hier | 0.98063 | 0.87990 | 0.87276 | 0.97622 | 0.03047 | True |
| IP | 8 | hpg_hier_junction | 0.99013 | 0.83891 | 0.84523 | 0.98369 | 0.02338 | True |
| IP | 8 | hpg_hier_junction1 | 0.96833 | 0.60168 | 0.86217 | 0.95116 | 0.03637 | True |

## Octamer: baseline vs octamer vs wDMPNN — seed-42 per-fold metrics

| target | fold | model | group_mean_r2 | delta_r2 | ordering | overall_r2 | overall_mae | finite |
|---|---|---|---|---|---|---|---|---|
| EA | 0 | hpg_hier | 0.92534 | 0.73356 | 0.74454 | 0.92411 | 0.10681 | True |
| EA | 0 | hpg_hier_octamer | 0.99476 | 0.70815 | 0.74405 | 0.99138 | 0.03545 | True |
| EA | 0 | wdmpnn | 0.76013 | 0.68363 | 0.79554 | 0.75982 | 0.22686 | True |
| EA | 1 | hpg_hier | 0.57514 | 0.79027 | 0.81215 | 0.56850 | 0.21397 | True |
| EA | 1 | hpg_hier_octamer | 0.98890 | 0.83369 | 0.86241 | 0.98434 | 0.03141 | True |
| EA | 1 | wdmpnn | 0.94464 | 0.43639 | 0.78837 | 0.93036 | 0.08001 | True |
| EA | 2 | hpg_hier | 0.92157 | 0.84550 | 0.80564 | 0.92365 | 0.11819 | True |
| EA | 2 | hpg_hier_octamer | 0.99464 | 0.85822 | 0.83781 | 0.99236 | 0.03478 | True |
| EA | 2 | wdmpnn | 0.98035 | 0.48507 | 0.70235 | 0.97263 | 0.06299 | True |
| EA | 3 | hpg_hier | 0.97464 | 0.83719 | 0.79896 | 0.96855 | 0.05469 | True |
| EA | 3 | hpg_hier_octamer | 0.98362 | 0.91279 | 0.87887 | 0.98052 | 0.04501 | True |
| EA | 3 | wdmpnn | 0.99032 | 0.72566 | 0.77680 | 0.98149 | 0.04095 | True |
| EA | 4 | hpg_hier | 0.95060 | 0.34695 | 0.57804 | 0.93451 | 0.07334 | True |
| EA | 4 | hpg_hier_octamer | 0.89700 | 0.13872 | 0.57209 | 0.87842 | 0.10833 | True |
| EA | 4 | wdmpnn | 0.94556 | 0.14535 | 0.51238 | 0.93114 | 0.07598 | True |
| EA | 5 | hpg_hier | 0.96903 | 0.88197 | 0.84669 | 0.96482 | 0.05831 | True |
| EA | 5 | hpg_hier_octamer | 0.99138 | 0.84676 | 0.83439 | 0.98633 | 0.03328 | True |
| EA | 5 | wdmpnn | 0.96868 | 0.58024 | 0.73428 | 0.95590 | 0.06451 | True |
| EA | 6 | hpg_hier | 0.91655 | 0.24291 | 0.79635 | 0.88211 | 0.05805 | True |
| EA | 6 | hpg_hier_octamer | 0.92480 | 0.28028 | 0.81777 | 0.89776 | 0.05652 | True |
| EA | 6 | wdmpnn | 0.89363 | 0.33894 | 0.64451 | 0.86504 | 0.06855 | True |
| EA | 7 | hpg_hier | 0.90187 | 0.64959 | 0.79472 | 0.90497 | 0.15554 | True |
| EA | 7 | hpg_hier_octamer | 0.98949 | 0.78890 | 0.78763 | 0.98826 | 0.05566 | True |
| EA | 7 | wdmpnn | 0.96479 | 0.58032 | 0.75301 | 0.96299 | 0.10081 | True |
| EA | 8 | hpg_hier | 0.96315 | 0.84228 | 0.85402 | 0.96150 | 0.09393 | True |
| EA | 8 | hpg_hier_octamer | 0.99396 | 0.78534 | 0.81826 | 0.99142 | 0.04256 | True |
| EA | 8 | wdmpnn | 0.99408 | 0.62748 | 0.79260 | 0.98944 | 0.04414 | True |
| IP | 0 | hpg_hier | 0.92787 | 0.82822 | 0.82682 | 0.92315 | 0.06460 | True |
| IP | 0 | hpg_hier_octamer | 0.92650 | 0.92093 | 0.86820 | 0.92311 | 0.06590 | True |
| IP | 0 | wdmpnn | 0.26951 | 0.71311 | 0.81623 | 0.26476 | 0.22411 | True |
| IP | 1 | hpg_hier | 0.98116 | 0.73747 | 0.82568 | 0.97898 | 0.04499 | True |
| IP | 1 | hpg_hier_octamer | 0.96971 | 0.76280 | 0.79480 | 0.96739 | 0.05792 | True |
| IP | 1 | wdmpnn | 0.96354 | 0.30306 | 0.74340 | 0.95362 | 0.06211 | True |
| IP | 2 | hpg_hier | 0.76920 | 0.21126 | 0.75692 | 0.77432 | 0.19782 | True |
| IP | 2 | hpg_hier_octamer | 0.99387 | 0.51657 | 0.65730 | 0.99095 | 0.03638 | True |
| IP | 2 | wdmpnn | 0.97667 | 0.42393 | 0.71733 | 0.97301 | 0.06592 | True |
| IP | 3 | hpg_hier | 0.97463 | 0.81947 | 0.82193 | 0.96311 | 0.03347 | True |
| IP | 3 | hpg_hier_octamer | 0.93263 | 0.88459 | 0.87529 | 0.92418 | 0.05369 | True |
| IP | 3 | wdmpnn | 0.96794 | 0.82517 | 0.86999 | 0.95694 | 0.03798 | True |
| IP | 4 | hpg_hier | 0.94379 | 0.58074 | 0.76393 | 0.93211 | 0.04514 | True |
| IP | 4 | hpg_hier_octamer | 0.91007 | 0.51872 | 0.80930 | 0.89351 | 0.05746 | True |
| IP | 4 | wdmpnn | 0.71193 | 0.44493 | 0.75774 | 0.68960 | 0.09734 | True |
| IP | 5 | hpg_hier | 0.76969 | 0.89601 | 0.91675 | 0.76798 | 0.11849 | True |
| IP | 5 | hpg_hier_octamer | 0.87641 | 0.94125 | 0.92204 | 0.87469 | 0.08144 | True |
| IP | 5 | wdmpnn | 0.94604 | 0.46005 | 0.82861 | 0.86816 | 0.07590 | True |
| IP | 6 | hpg_hier | 0.97462 | 0.87979 | 0.84620 | 0.97164 | 0.05101 | True |
| IP | 6 | hpg_hier_octamer | 0.98614 | 0.91168 | 0.82706 | 0.98415 | 0.03869 | True |
| IP | 6 | wdmpnn | 0.98380 | 0.38496 | 0.60264 | 0.96847 | 0.05419 | True |
| IP | 7 | hpg_hier | 0.96862 | 0.77267 | 0.73053 | 0.96628 | 0.06580 | True |
| IP | 7 | hpg_hier_octamer | 0.98848 | 0.85914 | 0.75684 | 0.98626 | 0.03965 | True |
| IP | 7 | wdmpnn | 0.98326 | 0.61033 | 0.66048 | 0.97779 | 0.04733 | True |
| IP | 8 | hpg_hier | 0.98063 | 0.87990 | 0.87276 | 0.97622 | 0.03047 | True |
| IP | 8 | hpg_hier_octamer | 0.98371 | 0.86802 | 0.87268 | 0.97751 | 0.02564 | True |
| IP | 8 | wdmpnn | 0.97688 | 0.77574 | 0.82812 | 0.96751 | 0.03397 | True |

## Seed-43/44 aggregation status

The requested three-seed aggregation and paired Wilcoxon tests require all 9 folds for each seed/model/target. This workspace has no complete seed-43 or seed-44 LOMO set for any requested model; no three-seed claim or p-value is reported.

