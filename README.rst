Dow Jones DNA Common (djdna_common)
###################################

Common methods for scripts and notebooks when working with `Dow Jones DNA data <https://developer.dowjones.com/site/global/develop/analytics_and_services/introduction/index.gsp>`_.

Contains useful methods for the following purposes:

* **Snapshot Files**: Contains methods to read snapshot files in all supported formats, from the DNA service.
* **Elasticsearch**: Methods to save sets of documents to an Elasticsearch server
* **Enrichment**: Methods to add features to a news articles dataset. Features range from simple calculations to vectors.
* **Visualisation**: Methods to ease the transformations of news articles dataset for visualisations in Jupyter Notebooks.

At some point (with a more stable version), this library may become available as a package.

In the meantime, this repository can be cloned and included in "Other Project" using a symbolic link. The following command sequence can be used as reference to create such symbolic link.

.. code-block::

    $ git clone https://github.com/miballe/djdna-other-project.git
    $ git clone https://github.com/miballe/djdna_common
    $ cd djdna-other-project
    $ ln -s ../djdna_common/ djdna_common
