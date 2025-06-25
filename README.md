# Towards a Perspectivist Turn in Argument Quality Assessment

## Database
An interactive version of the database is available under: https://goofy-grouse-1da.notion.site/AQ-DQ-Datasets-e3e5886191ef472aaffb47fec0daea92

The underlying xlsx file can be found in the *database folder*. The 32 columns of meta-information provided are described in detail in Appendix B of the paper.

Versions:
- v1.1: update D31

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
@inproceedings{romberg-etal-2025-towards,
    title = "Towards a Perspectivist Turn in Argument Quality Assessment",
    author = "Romberg, Julia  and
      Maurer, Maximilian  and
      Wachsmuth, Henning  and
      Lapesa, Gabriella",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.382/",
    doi = "10.18653/v1/2025.naacl-long.382",
    pages = "7458--7485",
    ISBN = "979-8-89176-189-6",
    abstract = "The assessment of argument quality depends on well-established logical, rhetorical, and dialectical properties that are unavoidably subjective: multiple valid assessments may exist, there is no unequivocal ground truth. This aligns with recent paths in machine learning, which embrace the co-existence of different perspectives. However, this potential remains largely unexplored in NLP research on argument quality. One crucial reason seems to be the yet unexplored availability of suitable datasets. We fill this gap by conducting a systematic review of argument quality datasets. We assign them to a multi-layered categorization targeting two aspects: (a) What has been annotated: we collect the quality dimensions covered in datasets and consolidate them in an overarching taxonomy, increasing dataset comparability and interoperability. (b) Who annotated: we survey what information is given about annotators, enabling perspectivist research and grounding our recommendations for future actions. To this end, we discuss datasets suitable for developing perspectivist models (i.e., those containing individual, non-aggregated annotations), and we showcase the importance of a controlled selection of annotators in a pilot study."
}
```
