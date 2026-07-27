# LOMO Multi-Seed Results

Metrics use the existing LOMO group key and matched-architecture definition. Metrics are averaged across seeds within each fold before fold medians/means and paired tests. Missing cells are explicit below. With nine folds, the minimum attainable exact two-sided sign-test p-value is 0.0039. Holm adjustment is across this complete comparison family.

## Inventory

Available cells: `72`; missing cells: `18`.

| model | target | fold | seed | path |
| --- | --- | --- | --- | --- |
| hpg_hier_attention | EA | 0 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold0__s42.npz |
| hpg_hier_attention | EA | 1 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold1__s42.npz |
| hpg_hier_attention | EA | 2 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold2__s42.npz |
| hpg_hier_attention | EA | 3 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold3__s42.npz |
| hpg_hier_attention | EA | 4 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold4__s42.npz |
| hpg_hier_attention | EA | 5 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold5__s42.npz |
| hpg_hier_attention | EA | 6 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold6__s42.npz |
| hpg_hier_attention | EA | 7 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold7__s42.npz |
| hpg_hier_attention | EA | 8 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__EA_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold8__s42.npz |
| hpg_hier_attention | IP | 0 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__IP_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold0__s42.npz |
| hpg_hier_attention | IP | 1 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__IP_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold1__s42.npz |
| hpg_hier_attention | IP | 2 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__IP_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold2__s42.npz |
| hpg_hier_attention | IP | 3 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__IP_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold3__s42.npz |
| hpg_hier_attention | IP | 4 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__IP_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold4__s42.npz |
| hpg_hier_attention | IP | 5 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__IP_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold5__s42.npz |
| hpg_hier_attention | IP | 6 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__IP_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold6__s42.npz |
| hpg_hier_attention | IP | 7 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__IP_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold7__s42.npz |
| hpg_hier_attention | IP | 8 | 42 | /Users/u6788552/Desktop/experiments/dmpnn/predictions/ea_ip_lomo/ea_ip__IP_vs_SHE_eV__hpg_hier_attention__monomer_heldout__fold8__s42.npz |

## Per-fold seed-averaged metrics

| model | target | fold | n_seeds | group_mean_r2 | delta_r2 | ordering | overall_r2 | mae | a_blind_null_group_mean_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpg_hier | EA | 0 | 1 | 0.92534 | 0.73356 | 0.74454 | 0.92411 | 0.10681 | 0.69380 |
| hpg_hier | EA | 1 | 1 | 0.57514 | 0.79027 | 0.81215 | 0.56850 | 0.21397 | 0.48731 |
| hpg_hier | EA | 2 | 1 | 0.92157 | 0.84550 | 0.80564 | 0.92365 | 0.11819 | 0.96114 |
| hpg_hier | EA | 3 | 1 | 0.97464 | 0.83719 | 0.79896 | 0.96855 | 0.05469 | 0.95276 |
| hpg_hier | EA | 4 | 1 | 0.95060 | 0.34695 | 0.57804 | 0.93451 | 0.07334 | 0.88373 |
| hpg_hier | EA | 5 | 1 | 0.96903 | 0.88197 | 0.84669 | 0.96482 | 0.05831 | 0.67571 |
| hpg_hier | EA | 6 | 1 | 0.91655 | 0.24291 | 0.79635 | 0.88211 | 0.05805 | -19.06946 |
| hpg_hier | EA | 7 | 1 | 0.90187 | 0.64959 | 0.79472 | 0.90497 | 0.15554 | 0.09751 |
| hpg_hier | EA | 8 | 1 | 0.96315 | 0.84228 | 0.85402 | 0.96150 | 0.09393 | 0.42772 |
| hpg_hier | IP | 0 | 1 | 0.92787 | 0.82822 | 0.82682 | 0.92315 | 0.06460 | 0.96902 |
| hpg_hier | IP | 1 | 1 | 0.98116 | 0.73747 | 0.82568 | 0.97898 | 0.04499 | 0.50927 |
| hpg_hier | IP | 2 | 1 | 0.76920 | 0.21126 | 0.75692 | 0.77432 | 0.19782 | -1.01943 |
| hpg_hier | IP | 3 | 1 | 0.97463 | 0.81947 | 0.82193 | 0.96311 | 0.03347 | -3.20636 |
| hpg_hier | IP | 4 | 1 | 0.94379 | 0.58074 | 0.76393 | 0.93211 | 0.04514 | -0.25097 |
| hpg_hier | IP | 5 | 1 | 0.76969 | 0.89601 | 0.91675 | 0.76798 | 0.11849 | -7.52773 |
| hpg_hier | IP | 6 | 1 | 0.97462 | 0.87979 | 0.84620 | 0.97164 | 0.05101 | 0.56868 |
| hpg_hier | IP | 7 | 1 | 0.96862 | 0.77267 | 0.73053 | 0.96628 | 0.06580 | 0.40980 |
| hpg_hier | IP | 8 | 1 | 0.98063 | 0.87990 | 0.87276 | 0.97622 | 0.03047 | -0.03401 |
| hpg_hier_junction | EA | 0 | 1 | 0.91531 | 0.77177 | 0.78804 | 0.91221 | 0.11850 | 0.69380 |
| hpg_hier_junction | EA | 1 | 1 | 0.92462 | 0.76452 | 0.72353 | 0.91801 | 0.08649 | 0.48731 |
| hpg_hier_junction | EA | 2 | 1 | 0.97848 | 0.83113 | 0.77533 | 0.97646 | 0.06545 | 0.96114 |
| hpg_hier_junction | EA | 3 | 1 | 0.98604 | 0.80348 | 0.86869 | 0.98004 | 0.04058 | 0.95276 |
| hpg_hier_junction | EA | 4 | 1 | 0.98012 | 0.56119 | 0.53552 | 0.97328 | 0.05222 | 0.88373 |
| hpg_hier_junction | EA | 5 | 1 | 0.97449 | 0.88365 | 0.84066 | 0.97176 | 0.04943 | 0.67571 |
| hpg_hier_junction | EA | 6 | 1 | 0.84377 | -0.68791 | 0.67286 | 0.76390 | 0.08893 | -19.06946 |
| hpg_hier_junction | EA | 7 | 1 | 0.96494 | 0.78790 | 0.81981 | 0.96476 | 0.09969 | 0.09751 |
| hpg_hier_junction | EA | 8 | 1 | 0.97099 | 0.80528 | 0.82095 | 0.96887 | 0.08970 | 0.42772 |
| hpg_hier_junction | IP | 0 | 1 | 0.98123 | 0.87665 | 0.83056 | 0.97650 | 0.03285 | 0.96902 |
| hpg_hier_junction | IP | 1 | 1 | 0.96175 | 0.77508 | 0.81753 | 0.96060 | 0.06280 | 0.50927 |
| hpg_hier_junction | IP | 2 | 1 | 0.87077 | 0.67182 | 0.76507 | 0.87333 | 0.15918 | -1.01943 |
| hpg_hier_junction | IP | 3 | 1 | 0.84529 | 0.73310 | 0.80938 | 0.83291 | 0.08116 | -3.20636 |
| hpg_hier_junction | IP | 4 | 1 | 0.92729 | 0.59595 | 0.80108 | 0.91982 | 0.04954 | -0.25097 |
| hpg_hier_junction | IP | 5 | 1 | 0.49409 | 0.76665 | 0.88025 | 0.49896 | 0.17561 | -7.52773 |
| hpg_hier_junction | IP | 6 | 1 | 0.99305 | 0.80191 | 0.80531 | 0.98786 | 0.03155 | 0.56868 |
| hpg_hier_junction | IP | 7 | 1 | 0.98008 | 0.83088 | 0.74845 | 0.97801 | 0.04767 | 0.40980 |
| hpg_hier_junction | IP | 8 | 1 | 0.99013 | 0.83891 | 0.84523 | 0.98369 | 0.02338 | -0.03401 |
| hpg_hier_octamer | EA | 0 | 1 | 0.99476 | 0.70815 | 0.74405 | 0.99138 | 0.03545 | 0.69380 |
| hpg_hier_octamer | EA | 1 | 1 | 0.98890 | 0.83369 | 0.86217 | 0.98434 | 0.03141 | 0.48731 |
| hpg_hier_octamer | EA | 2 | 1 | 0.99464 | 0.85822 | 0.83757 | 0.99236 | 0.03478 | 0.96114 |
| hpg_hier_octamer | EA | 3 | 1 | 0.98362 | 0.91279 | 0.87862 | 0.98052 | 0.04501 | 0.95276 |
| hpg_hier_octamer | EA | 4 | 1 | 0.89700 | 0.13872 | 0.57185 | 0.87842 | 0.10833 | 0.88373 |
| hpg_hier_octamer | EA | 5 | 1 | 0.99138 | 0.84676 | 0.83415 | 0.98633 | 0.03328 | 0.67571 |
| hpg_hier_octamer | EA | 6 | 1 | 0.92480 | 0.28028 | 0.81769 | 0.89776 | 0.05652 | -19.06946 |
| hpg_hier_octamer | EA | 7 | 1 | 0.98949 | 0.78890 | 0.78739 | 0.98826 | 0.05566 | 0.09751 |
| hpg_hier_octamer | EA | 8 | 1 | 0.99396 | 0.78534 | 0.81818 | 0.99142 | 0.04256 | 0.42772 |
| hpg_hier_octamer | IP | 0 | 1 | 0.92650 | 0.92093 | 0.86820 | 0.92311 | 0.06590 | 0.96902 |
| hpg_hier_octamer | IP | 1 | 1 | 0.96971 | 0.76280 | 0.79472 | 0.96739 | 0.05792 | 0.50927 |
| hpg_hier_octamer | IP | 2 | 1 | 0.99387 | 0.51657 | 0.65705 | 0.99095 | 0.03638 | -1.01943 |
| hpg_hier_octamer | IP | 3 | 1 | 0.93263 | 0.88459 | 0.87520 | 0.92418 | 0.05369 | -3.20636 |
| hpg_hier_octamer | IP | 4 | 1 | 0.91007 | 0.51872 | 0.80922 | 0.89351 | 0.05746 | -0.25097 |
| hpg_hier_octamer | IP | 5 | 1 | 0.87641 | 0.94125 | 0.92196 | 0.87469 | 0.08144 | -7.52773 |
| hpg_hier_octamer | IP | 6 | 1 | 0.98614 | 0.91168 | 0.82698 | 0.98415 | 0.03869 | 0.56868 |
| hpg_hier_octamer | IP | 7 | 1 | 0.98848 | 0.85914 | 0.75660 | 0.98626 | 0.03965 | 0.40980 |
| hpg_hier_octamer | IP | 8 | 1 | 0.98371 | 0.86802 | 0.87243 | 0.97751 | 0.02564 | -0.03401 |
| wdmpnn | EA | 0 | 1 | 0.76013 | 0.68363 | 0.79554 | 0.75982 | 0.22686 | 0.69380 |
| wdmpnn | EA | 1 | 1 | 0.94464 | 0.43639 | 0.78837 | 0.93036 | 0.08001 | 0.48731 |
| wdmpnn | EA | 2 | 1 | 0.98035 | 0.48507 | 0.70235 | 0.97263 | 0.06299 | 0.96114 |
| wdmpnn | EA | 3 | 1 | 0.99032 | 0.72566 | 0.77680 | 0.98149 | 0.04095 | 0.95276 |
| wdmpnn | EA | 4 | 1 | 0.94556 | 0.14535 | 0.51238 | 0.93114 | 0.07598 | 0.88373 |
| wdmpnn | EA | 5 | 1 | 0.96868 | 0.58024 | 0.73428 | 0.95590 | 0.06451 | 0.67571 |
| wdmpnn | EA | 6 | 1 | 0.89363 | 0.33894 | 0.64451 | 0.86504 | 0.06855 | -19.06946 |
| wdmpnn | EA | 7 | 1 | 0.96479 | 0.58032 | 0.75301 | 0.96299 | 0.10081 | 0.09751 |
| wdmpnn | EA | 8 | 1 | 0.99408 | 0.62748 | 0.79260 | 0.98944 | 0.04414 | 0.42772 |
| wdmpnn | IP | 0 | 1 | 0.26951 | 0.71311 | 0.81623 | 0.26476 | 0.22411 | 0.96902 |
| wdmpnn | IP | 1 | 1 | 0.96354 | 0.30306 | 0.74340 | 0.95362 | 0.06211 | 0.50927 |
| wdmpnn | IP | 2 | 1 | 0.97667 | 0.42393 | 0.71733 | 0.97301 | 0.06592 | -1.01943 |
| wdmpnn | IP | 3 | 1 | 0.96794 | 0.82517 | 0.86999 | 0.95694 | 0.03798 | -3.20636 |
| wdmpnn | IP | 4 | 1 | 0.71193 | 0.44493 | 0.75774 | 0.68960 | 0.09734 | -0.25097 |
| wdmpnn | IP | 5 | 1 | 0.94604 | 0.46005 | 0.82861 | 0.86816 | 0.07590 | -7.52773 |
| wdmpnn | IP | 6 | 1 | 0.98380 | 0.38496 | 0.60264 | 0.96847 | 0.05419 | 0.56868 |
| wdmpnn | IP | 7 | 1 | 0.98326 | 0.61033 | 0.66048 | 0.97779 | 0.04733 | 0.40980 |
| wdmpnn | IP | 8 | 1 | 0.97688 | 0.77574 | 0.82812 | 0.96751 | 0.03397 | -0.03401 |

## Across-fold summary

| model | target | group_mean_r2_median | group_mean_r2_mean | delta_r2_median | delta_r2_mean | ordering_median | ordering_mean | overall_r2_median | overall_r2_mean | mae_median | mae_mean | fold_bias_median | fold_bias_mean | compression_ratio_median | compression_ratio_mean | a_blind_null_group_mean_r2_median | a_blind_null_group_mean_r2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpg_hier | EA | 0.92534 | 0.89976 | 0.79027 | 0.68558 | 0.79896 | 0.78124 | 0.92411 | 0.89252 | 0.09393 | 0.10365 | -0.04163 | -0.03213 | 1.08458 | 1.03368 | 0.67571 | -1.54331 |
| hpg_hier | IP | 0.96862 | 0.92113 | 0.81947 | 0.73395 | 0.82568 | 0.81795 | 0.96311 | 0.91709 | 0.05101 | 0.07242 | -0.00767 | -0.01203 | 0.99560 | 0.99245 | -0.03401 | -1.06464 |
| hpg_hier_junction | EA | 0.97099 | 0.94875 | 0.78790 | 0.61345 | 0.78804 | 0.76060 | 0.96887 | 0.93659 | 0.08649 | 0.07678 | -0.00962 | -0.00481 | 0.99979 | 1.02420 | 0.67571 | -1.54331 |
| hpg_hier_junction | IP | 0.96175 | 0.89374 | 0.77508 | 0.76566 | 0.80938 | 0.81143 | 0.96060 | 0.89019 | 0.04954 | 0.07375 | -0.02840 | -0.01916 | 1.00540 | 1.01428 | -0.03401 | -1.06464 |
| hpg_hier_octamer | EA | 0.98949 | 0.97317 | 0.78890 | 0.68365 | 0.81818 | 0.79463 | 0.98633 | 0.96565 | 0.04256 | 0.04922 | -0.00925 | -0.01601 | 1.01292 | 0.99237 | 0.67571 | -1.54331 |
| hpg_hier_octamer | IP | 0.96971 | 0.95195 | 0.86802 | 0.79819 | 0.82698 | 0.82026 | 0.96739 | 0.94686 | 0.05369 | 0.05075 | 0.03012 | 0.02451 | 1.01271 | 1.00629 | -0.03401 | -1.06464 |
| wdmpnn | EA | 0.96479 | 0.93802 | 0.58024 | 0.51145 | 0.75301 | 0.72220 | 0.95590 | 0.92765 | 0.06855 | 0.08498 | -0.02137 | -0.01998 | 1.00210 | 1.00846 | 0.67571 | -1.54331 |
| wdmpnn | IP | 0.96794 | 0.86440 | 0.46005 | 0.54903 | 0.75774 | 0.75828 | 0.95694 | 0.84665 | 0.06211 | 0.07765 | 0.03149 | 0.04877 | 1.05020 | 1.06540 | -0.03401 | -1.06464 |

## Pooled placement metrics

| model | target | pooled_group_mean_r2 | fold_placement_r2 | fold_placement_slope | fold_placement_intercept | fold_bias_sd | n_available_fold_seed_cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpg_hier | EA | 0.94887 | 0.92306 | 0.91292 | -0.25344 | 0.10273 | 9 |
| hpg_hier | IP | 0.95187 | 0.93921 | 0.80603 | 0.26989 | 0.08316 | 9 |
| wdmpnn | EA | 0.96653 | 0.94550 | 0.99427 | -0.03455 | 0.08836 | 9 |
| wdmpnn | IP | 0.95406 | 0.93604 | 1.00256 | 0.04506 | 0.07106 | 9 |
| hpg_hier_octamer | EA | 0.98907 | 0.98941 | 1.03069 | 0.06198 | 0.03658 | 9 |
| hpg_hier_octamer | IP | 0.98444 | 0.98013 | 1.05093 | -0.04952 | 0.04131 | 9 |
| hpg_hier_junction | EA | 0.97586 | 0.97105 | 0.89190 | -0.27952 | 0.06585 | 9 |
| hpg_hier_junction | IP | 0.95637 | 0.93658 | 0.81603 | 0.24821 | 0.08365 | 9 |

## Paired comparisons

| model | reference | target | metric | wins | losses | ties | sign_test_p | wilcoxon_p | holm_wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wdmpnn | hpg_hier | EA | group_mean_r2 | 5 | 4 | 0 | 1.00000 | 0.42578 | 1.00000 |
| wdmpnn | hpg_hier | EA | delta_r2 | 1 | 8 | 0 | 0.03906 | 0.01953 | 0.58594 |
| wdmpnn | hpg_hier | EA | ordering | 1 | 8 | 0 | 0.03906 | 0.02734 | 0.79297 |
| wdmpnn | hpg_hier | EA | overall_r2 | 5 | 4 | 0 | 1.00000 | 0.42578 | 1.00000 |
| wdmpnn | hpg_hier | EA | mae | 5 | 4 | 0 | 1.00000 | 0.35938 | 1.00000 |
| hpg_hier_octamer | hpg_hier | EA | group_mean_r2 | 8 | 1 | 0 | 0.03906 | 0.03906 | 1.00000 |
| hpg_hier_octamer | hpg_hier | EA | delta_r2 | 5 | 4 | 0 | 1.00000 | 0.82031 | 1.00000 |
| hpg_hier_octamer | hpg_hier | EA | ordering | 4 | 5 | 0 | 1.00000 | 0.57031 | 1.00000 |
| hpg_hier_octamer | hpg_hier | EA | overall_r2 | 8 | 1 | 0 | 0.03906 | 0.03906 | 1.00000 |
| hpg_hier_octamer | hpg_hier | EA | mae | 8 | 1 | 0 | 0.03906 | 0.02734 | 0.79297 |
| hpg_hier_junction | hpg_hier | EA | group_mean_r2 | 7 | 2 | 0 | 0.17969 | 0.20312 | 1.00000 |
| hpg_hier_junction | hpg_hier | EA | delta_r2 | 4 | 5 | 0 | 1.00000 | 1.00000 | 1.00000 |
| hpg_hier_junction | hpg_hier | EA | ordering | 3 | 6 | 0 | 0.50781 | 0.42578 | 1.00000 |
| hpg_hier_junction | hpg_hier | EA | overall_r2 | 7 | 2 | 0 | 0.17969 | 0.25000 | 1.00000 |
| hpg_hier_junction | hpg_hier | EA | mae | 7 | 2 | 0 | 0.17969 | 0.12891 | 1.00000 |
| hpg_hier_attention | hpg_hier | EA | group_mean_r2 | 0 | 0 | 9 | 1.00000 | nan | nan |
| hpg_hier_attention | hpg_hier | EA | delta_r2 | 0 | 0 | 9 | 1.00000 | nan | nan |
| hpg_hier_attention | hpg_hier | EA | ordering | 0 | 0 | 9 | 1.00000 | nan | nan |
| hpg_hier_attention | hpg_hier | EA | overall_r2 | 0 | 0 | 9 | 1.00000 | nan | nan |
| hpg_hier_attention | hpg_hier | EA | mae | 0 | 0 | 9 | 1.00000 | nan | nan |
| wdmpnn | hpg_hier | IP | group_mean_r2 | 4 | 5 | 0 | 1.00000 | 0.82031 | 1.00000 |
| wdmpnn | hpg_hier | IP | delta_r2 | 2 | 7 | 0 | 0.17969 | 0.07422 | 1.00000 |
| wdmpnn | hpg_hier | IP | ordering | 1 | 8 | 0 | 0.03906 | 0.03906 | 1.00000 |
| wdmpnn | hpg_hier | IP | overall_r2 | 3 | 6 | 0 | 0.50781 | 0.57031 | 1.00000 |
| wdmpnn | hpg_hier | IP | mae | 3 | 6 | 0 | 0.50781 | 0.73438 | 1.00000 |
| hpg_hier_octamer | hpg_hier | IP | group_mean_r2 | 5 | 4 | 0 | 1.00000 | 0.57031 | 1.00000 |
| hpg_hier_octamer | hpg_hier | IP | delta_r2 | 7 | 2 | 0 | 0.17969 | 0.05469 | 1.00000 |
| hpg_hier_octamer | hpg_hier | IP | ordering | 5 | 4 | 0 | 1.00000 | 0.65234 | 1.00000 |
| hpg_hier_octamer | hpg_hier | IP | overall_r2 | 5 | 4 | 0 | 1.00000 | 0.57031 | 1.00000 |
| hpg_hier_octamer | hpg_hier | IP | mae | 5 | 4 | 0 | 1.00000 | 0.42578 | 1.00000 |
| hpg_hier_junction | hpg_hier | IP | group_mean_r2 | 5 | 4 | 0 | 1.00000 | 0.82031 | 1.00000 |
| hpg_hier_junction | hpg_hier | IP | delta_r2 | 5 | 4 | 0 | 1.00000 | 0.91016 | 1.00000 |
| hpg_hier_junction | hpg_hier | IP | ordering | 4 | 5 | 0 | 1.00000 | 0.49609 | 1.00000 |
| hpg_hier_junction | hpg_hier | IP | overall_r2 | 5 | 4 | 0 | 1.00000 | 0.82031 | 1.00000 |
| hpg_hier_junction | hpg_hier | IP | mae | 5 | 4 | 0 | 1.00000 | 0.91016 | 1.00000 |
| hpg_hier_attention | hpg_hier | IP | group_mean_r2 | 0 | 0 | 9 | 1.00000 | nan | nan |
| hpg_hier_attention | hpg_hier | IP | delta_r2 | 0 | 0 | 9 | 1.00000 | nan | nan |
| hpg_hier_attention | hpg_hier | IP | ordering | 0 | 0 | 9 | 1.00000 | nan | nan |
| hpg_hier_attention | hpg_hier | IP | overall_r2 | 0 | 0 | 9 | 1.00000 | nan | nan |
| hpg_hier_attention | hpg_hier | IP | mae | 0 | 0 | 9 | 1.00000 | nan | nan |
