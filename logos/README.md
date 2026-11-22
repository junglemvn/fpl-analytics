# Club emblems

This folder is the only logo source used by Notebook 2. Its 27 PNG files cover all clubs
represented in the prepared 2022/23–2026/27 data, including historical relegated clubs.
Keep the existing filenames; `src/plot_annotations.py` maps the FPL display names to them.
The historical name `Ipswich` and current name `Ipswich Town` use the same image.

The notebook validates club coverage separately for each season. Its EDA audit records
`season`, `fpl_team_id` and local asset path, while the manifest checksums each distinct PNG.
Existing emblems are preserved byte-for-byte; no image downloads or external logo archive
are needed when rerunning the notebook.
