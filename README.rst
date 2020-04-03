Dow Jones Factiva Common (factiva_common)
#########################################

Common methods for scripts and notebooks when working with `Dow Jones Factiva Snapshots & Streams data <https://developer.dowjones.com/site/docs/factiva_apis/factiva_snapshots_api/index.gsp>`_.

Contains useful methods for the following purposes:

* **Snapshot Files**: Contains methods to read snapshot files in supported formats (initially AVRO).
* **Elasticsearch**: Methods to save sets of documents to an Elasticsearch server
* **Enrichment**: Methods to add features to a news articles dataset. Features range from simple calculations to vectors.
* **Visualisation**: Methods to ease the transformations of news articles dataset for visualisations in Jupyter Notebooks.

At some point (with a more stable version), this library may become available as a package.

In the meantime, this repository can be cloned and "included" in other projects using a symbolic link. The following command sequence can be used as reference to create such symbolic link.

.. code-block::

    $ git clone https://github.com/dowjones/factiva-other-project.git
    $ git clone https://github.com/dowjones/factiva_common.git
    $ cd factiva-other-project
    $ ln -s ../factiva_common/ factiva_common
