# Training data

`tg_training_data.csv` is a two-column extract (PSMILES, Tg in K) of 7,367
polymers from:

> Kunchapu, S.; Jablonka, K. M. *Curated Glass Transition Temperature for
> Polymers.* Zenodo. https://doi.org/10.5281/zenodo.14980914

That record is published under the **Creative Commons Attribution 4.0
International** licence, which permits redistribution with attribution. The
extract is included here so that the regression test which reproduces the
reported metrics can run in a bare checkout and in continuous integration,
rather than skipping.

The full curated record, including the provenance, reliability flags and
descriptor columns that this extract drops, should be obtained from the Zenodo
DOI above.
