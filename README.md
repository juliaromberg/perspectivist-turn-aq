# Towards a Perspectivist Turn in Argument Quality Assessment

## Database
An interactive version of the database is available under: https://goofy-grouse-1da.notion.site/AQ-DQ-Datasets-e3e5886191ef472aaffb47fec0daea92

The underlying xlsx file can be found in the *database folder*. The 32 columns of meta-information provided are described in detail in Appendix B of the paper.

## Experiments
All code can be found in the *database folder*.

To reproduce our analyses, request the raw data from the respective original authors.

Preprocess the raw data (see [code/expected_format.md](code/expected_format.md) for details on the expected data).

To do so, install the requirements
```bash
python3 -m pip install -r requirements.txt
```

and run the preprocessing scripts:
```bash
python3 preprocess_dagstuhl.py --path path/to/expert/csv --output output/directory --annotation_type expert
python3 preprocess_dagstuhl.py --path path/to/crowd/csv --output output/directory --annotation_type crowd
python3 preprocess_dagstuhl.py --path path/to/novice/directory --output output/directory --annotation_type novice
python3 preprocess_gaq.py --path path/to/GAQ/directory --ouput output/directory
```
The analyses done on the preprocessed data can be found in [code/analysis.ipynb](code/analysis.ipynb).

## Reference
```bibtex
TODO
```
