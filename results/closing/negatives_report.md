Loading weights:   0%|          | 0/203 [00:00<?, ?it/s]Loading weights:  12%|█▏        | 25/203 [00:00<00:00, 229.26it/s]Loading weights:  24%|██▎       | 48/203 [00:00<00:00, 188.08it/s]Loading weights:  37%|███▋      | 76/203 [00:00<00:00, 220.32it/s]Loading weights: 100%|██████████| 203/203 [00:00<00:00, 561.44it/s]
## Screen of the provenance clips at 1000 ms with AST (AudioSet)

|  | speech clips | non-speech clips |
| --- | --- | --- |
| 0.05 | p_speech | 0.689 | 0.006 |
| 0.05 | p_singing | 0.0 | 0.0 |
| 0.05 | p_music | 0.005 | 0.003 |
| 0.5 | p_speech | 0.834 | 0.049 |
| 0.5 | p_singing | 0.0 | 0.001 |
| 0.5 | p_music | 0.011 | 0.025 |
| 0.95 | p_speech | 0.905 | 0.798 |
| 0.95 | p_singing | 0.003 | 0.026 |
| 0.95 | p_music | 0.46 | 0.832 | 

- non-speech clips with p_speech >= 0.1: 45 of 114, with p_singing >= 0.1: 1
- non-speech clips with p_speech >= 0.3: 26 of 114, with p_singing >= 0.3: 1
- non-speech clips with p_speech >= 0.5: 18 of 114, with p_singing >= 0.5: 0

### Non-speech clips to listen to first (p_speech or p_singing >= 0.1), most doubtful first

| base_id | start_s | p_speech | p_singing | p_music | top3 |
| --- | --- | --- | --- | --- | --- |
| voxconverse_rxulz_0103 | 667.8 | 0.85 | 0.01 | 0.03 | Speech 0.85; Male speech, man speaking 0.20; Outside, urban or manmade 0.03 |
| voxconverse_qvtia_0145 | 178.6 | 0.84 | 0.0 | 0.01 | Speech 0.84; Male speech, man speaking 0.08; Rustling leaves 0.04 |
| voxconverse_iiprr_0039 | 815.5 | 0.84 | 0.0 | 0.01 | Speech 0.84; Male speech, man speaking 0.32; Narration, monologue 0.06 |
| voxconverse_djqif_0147 | 360.8 | 0.84 | 0.0 | 0.01 | Speech 0.84; Male speech, man speaking 0.34; Narration, monologue 0.08 |
| voxconverse_rarij_0133 | 177.2 | 0.83 | 0.01 | 0.12 | Speech 0.83; Music 0.12; Male speech, man speaking 0.10 |
| voxconverse_epdpg_0011 | 284.3 | 0.82 | 0.0 | 0.0 | Speech 0.82; Door 0.15; Sliding door 0.05 |
| voxconverse_aorju_0047 | 386.3 | 0.79 | 0.0 | 0.02 | Speech 0.79; Male speech, man speaking 0.09; Inside, public space 0.09 |
| voxconverse_kiadt_0055 | 368.5 | 0.76 | 0.0 | 0.02 | Speech 0.76; Vehicle 0.02; Music 0.02 |
| voxconverse_jwggf_0057 | 997.7 | 0.75 | 0.0 | 0.03 | Speech 0.75; Bird 0.04; Coo 0.04 |
| voxconverse_qoarn_0101 | 840.0 | 0.75 | 0.0 | 0.01 | Speech 0.75; Crowd 0.20; Outside, urban or manmade 0.07 |
| voxconverse_jsymf_0113 | 80.5 | 0.74 | 0.0 | 0.01 | Speech 0.74; Zipper (clothing) 0.16; Single-lens reflex camera 0.05 |
| voxconverse_thnuq_0187 | 78.8 | 0.73 | 0.04 | 0.03 | Speech 0.73; Male speech, man speaking 0.04; Mantra 0.04 |
| voxconverse_xggbk_0193 | 957.6 | 0.71 | 0.0 | 0.02 | Speech 0.71; Male speech, man speaking 0.05; Crowd 0.05 |
| voxconverse_lhuly_0119 | 594.6 | 0.62 | 0.02 | 0.15 | Speech 0.62; Music 0.15; Whoop 0.12 |
| voxconverse_aisvi_0029 | 18.4 | 0.6 | 0.0 | 0.04 | Speech 0.60; Male speech, man speaking 0.09; Chopping (food) 0.07 |
| voxconverse_hgdez_0217 | 950.5 | 0.59 | 0.0 | 0.73 | Music 0.73; Speech 0.59; Ringtone 0.16 |
| voxconverse_fzwtp_0093 | 216.3 | 0.54 | 0.0 | 0.0 | Speech 0.54; Slam 0.20; Slap, smack 0.09 |
| voxconverse_ktvto_0069 | 31.9 | 0.54 | 0.0 | 0.02 | Speech 0.54; Whoop 0.05; Slap, smack 0.04 |
| voxconverse_uhfrw_0179 | 85.3 | 0.5 | 0.0 | 0.0 | Speech 0.50; Throat clearing 0.30; Male speech, man speaking 0.07 |
| voxconverse_nqcpi_0167 | 70.9 | 0.49 | 0.0 | 0.07 | Speech 0.49; Music 0.07; Sound effect 0.05 |
| voxconverse_jyirt_0065 | 25.5 | 0.47 | 0.0 | 0.02 | Ding 0.52; Speech 0.47; Clang 0.17 |
| voxconverse_fpfvy_0191 | 99.8 | 0.02 | 0.45 | 0.15 | Choir 0.45; Music 0.15; Singing 0.12 |
| voxconverse_optsn_0165 | 552.9 | 0.41 | 0.0 | 0.02 | Applause 0.58; Speech 0.41; Male speech, man speaking 0.05 |
| voxconverse_ytula_0085 | 290.2 | 0.39 | 0.0 | 0.01 | Speech 0.39; Vehicle 0.10; Slap, smack 0.08 |
| voxconverse_eqsta_0033 | 62.2 | 0.38 | 0.0 | 0.01 | Applause 0.44; Speech 0.38; Slosh 0.10 |
| voxconverse_xvxwv_0087 | 86.7 | 0.35 | 0.0 | 0.16 | Speech 0.35; Music 0.16; Television 0.03 |
| voxconverse_tucrg_0207 | 10.3 | 0.32 | 0.0 | 0.03 | Speech 0.32; Applause 0.22; Crowd 0.05 |
| voxconverse_udjij_0223 | 14.3 | 0.29 | 0.0 | 0.07 | Speech 0.29; Cheering 0.17; Crowd 0.10 |
| voxconverse_cawnd_0215 | 514.8 | 0.28 | 0.04 | 0.08 | Whoop 0.31; Speech 0.28; Bellow 0.21 |
| voxconverse_cjfer_0121 | 23.7 | 0.22 | 0.0 | 0.03 | Speech 0.22; Coo 0.15; Bird 0.12 |
| voxconverse_wwvcs_0015 | 938.0 | 0.22 | 0.0 | 0.01 | Speech 0.22; Snicker 0.10; Chuckle, chortle 0.06 |
| voxconverse_ljpes_0007 | 165.3 | 0.21 | 0.0 | 0.01 | Quack 0.71; Duck 0.40; Honk 0.21 |
| voxconverse_vmbga_0175 | 608.1 | 0.21 | 0.0 | 0.0 | Speech 0.21; Fart 0.04; Boing 0.03 |
| voxconverse_ztzzr_0131 | 150.4 | 0.18 | 0.0 | 0.0 | Speech 0.18; Chink, clink 0.17; Ding 0.14 |
| voxconverse_iacod_0051 | 640.7 | 0.15 | 0.0 | 0.01 | Snicker 0.21; Speech 0.15; Chuckle, chortle 0.11 |
| voxconverse_qjgpl_0075 | 71.6 | 0.15 | 0.0 | 0.02 | Speech 0.15; Vehicle 0.08; Sliding door 0.06 |
| voxconverse_kzmyi_0097 | 599.5 | 0.15 | 0.0 | 0.02 | Applause 0.85; Speech 0.15; Sound effect 0.03 |
| voxconverse_leneg_0021 | 33.5 | 0.15 | 0.01 | 0.67 | Music 0.67; Speech 0.15; Electronic music 0.07 |
| voxconverse_wdjyj_0025 | 81.8 | 0.14 | 0.0 | 0.1 | Speech 0.14; Crowd 0.11; Music 0.10 |
| voxconverse_bpzsc_0185 | 877.3 | 0.13 | 0.0 | 0.01 | Speech 0.13; Applause 0.12; Clip-clop 0.11 |
| voxconverse_xmyyy_0139 | 218.2 | 0.13 | 0.0 | 0.02 | Typewriter 0.22; Speech 0.13; Slap, smack 0.04 |
| voxconverse_qlrry_0163 | 432.2 | 0.13 | 0.06 | 0.57 | Music 0.57; Speech 0.13; Slap, smack 0.06 |
| voxconverse_plbbw_0019 | 80.5 | 0.13 | 0.0 | 0.01 | Cheering 0.44; Crowd 0.23; Speech 0.13 |
| voxconverse_rsypp_0153 | 990.5 | 0.13 | 0.05 | 0.04 | Bellow 0.18; Yell 0.13; Speech 0.13 |
| voxconverse_gfneh_0017 | 67.4 | 0.13 | 0.01 | 0.28 | Music 0.28; Speech 0.13; Crowd 0.10 |
| voxconverse_myjoe_0037 | 398.5 | 0.1 | 0.0 | 0.01 | Snicker 0.18; Speech 0.10; Chuckle, chortle 0.08 | 

