# PD-UniST: Prompt-Driven Universal Model for Unpaired H&E-to-IHC Stain Translation

This repository is the official implementation of the MICCAI 2025 paper, "PD-UniST: Prompt-Driven Universal Model for Unpaired H&E-to-IHC Stain Translation".

## Abstract

Conventional Hematoxylin-Eosin (H&E) staining is limited to revealing cell morphology and distribution, whereas Immunohistochemical (IHC) staining provides precise and specific visualization of protein activation at the molecular level. Virtual staining technology has emerged as a solution for highly efficient IHC examination, which directly transforms H&E-stained images into IHC-stained images. However, virtual staining is challenged by the insufficient mining of pathological semantics and the spatial misalignment of pathological semantics. In this paper, we propose PD-UniST, a prompt-driven universal model for unpaired H&E-to-IHC stain translation. Our proposed method utilizes a Protein-Aware Learning Strategy (PALS) with a Focal Optical Density (FOD) map to extract molecular-level pathological semantics, constraining the protein expression level between the generated image and the label. Furthermore, we introduce a Pathological Semantics-Preserving (PSP) module to enhance the model's learning of pathological semantics. Extensive experiments on the BCI and MIST-her2 datasets have demonstrated that PD-UniST effectively preserves pathological semantics and improves staining performance without requiring additional annotations.

## Dataset

* **Data Link:** [Google Drive](https://drive.google.com/drive/folders/1rn9BgbaqwkijbvLSm3pmv8GMF67wed4r?usp=sharing)

