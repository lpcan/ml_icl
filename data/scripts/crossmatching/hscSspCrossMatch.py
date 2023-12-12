#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2019 National Institutes of Natural Sciences
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

VERSION = "2023.05.16.1"

import numpy
pyfits = None # imported on demand

import collections
import itertools
import re

if bytes is str:
    # if python2
    zip = itertools.izip

def get_argparser():
    """
    @return argparse.ArgumentParser:
        the command line parser for this script.
    """
    import argparse

    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""
            Positionally cross-match a catalog to the HSC database.
            This script reads a catalog to generate an SQL query, which will be output to stdout.
            The output query can be posted to the DB server by means of hscReleaseQuery.py
            (../hscReleaseQuery/hscReleaseQuery.py).
            Thus: %(prog)s catalog.fits | ../hscReleaseQuery/hscReleaseQuery.py --user=USER@local -
        """
        )

    parser.add_argument('catalog_path', help="""
        Catalog file to match to the HSC catalog. This may be either
        a FITS catalog or numpy's npy file.  Reading from text is not yet implemented.
    """)

    parser.add_argument('--rerun', default="pdr3_wide", help="Rerun name.")

    parser.add_argument("--template", default=":embedded:", help="Path to a file containing a template SQL.  To see the embedded (default) one, use --show-template.")
    parser.add_argument("--show-template", action="store_true", help="Show the template SQL and exit.")

    parser.add_argument("--columns", nargs="*", default=[], help="Supplementary columns to get.")

    parser.add_argument("--ra", default="ra", help="R.A. column name of your catalog.")
    parser.add_argument("--dec", default="dec", help="Dec. column name of your catalog.")

    parser.add_argument("--accuracy", type=float, default=0.5, help="accuracy of positional match (in arcseconds)")

    return parser


def startup():
    """
    Entry point when not imported but executed
    """
    args = get_argparser().parse_args()
    main(**vars(args))


def main(
    catalog_path,
    rerun           = "pdr3_wide",
    template        = ":embedded:",
    show_template   = False,
    columns         = [],
    ra              = "ra",
    dec             = "dec",
    accuracy        = 0.5
):
    """
    Read a catalog and print out an SQL query string that cross-matches
    the catalog to the HSC database.
    """
    template = load_query_template(template)
    if show_template:
        print(template)
        return

    extra_columns = ','.join(quote_identifier(col) for col in columns)
    if extra_columns:
        extra_columns = ',' + extra_columns

    if type(catalog_path) == str:
        user_catalog = load_catalog(catalog_path)
    else:
        user_catalog = catalog_path
    if len(user_catalog) == 0:
        raise RuntimeError("No data in user's catalog.")

    cols = user_catalog.dtype.names
    quoted_cols = [quote_identifier("user."+col) for col in cols]
    cols_to_quoted = dict(zip(cols, quoted_cols))

    try:
        user_ra  = cols_to_quoted[ra]
        user_dec = cols_to_quoted[dec]
    except KeyError:
        raise RuntimeError("Columns --ra={ra} --dec={dec} not found in {user_catalog.dtype.names!s}".format(**locals()))

    formats  = [get_format (user_catalog[name].dtype.type) for name in user_catalog.dtype.names]
    sqltypes = [get_sqltype(user_catalog[name].dtype.type) for name in user_catalog.dtype.names]

    header = "user_catalog(" + ",".join(quoted_cols) + ") AS (VALUES"
    footer = ")"
    first = "(" + ",".join("'{}'::{}".format(f,t) for f,t in zip(formats, sqltypes)) + ")"
    others = ",(" + ",".join("'{}'".format(f) for f in formats) + ")"

    user_catalog = header + (first + others*(len(user_catalog)-1) + footer) % tuple(
        itertools.chain.from_iterable(user_catalog)
    )

    args = dict(
        rerun=quote_identifier(rerun),
        user_catalog=user_catalog,
        ra=user_ra,
        dec=user_dec,
        accuracy=accuracy,
        columns=extra_columns,
    )

    query = template.format(**args)
    print(query)
    return query


def load_catalog(catalog_path):
    """
    Load a catalog of any format.
    @param catalog_path: str
    @return record array
    """
    lowered_path = catalog_path.lower()

    if lowered_path.endswith(".fits") or lowered_path.endswith(".fits.gz"):
        return load_catalog_fits(catalog_path)
    if lowered_path.endswith(".npy"):
        return load_catalog_npy(catalog_path)

    return load_catalog_txt(catalog_path)


def load_catalog_fits(catalog_path):
    """
    Load a FITS catalog.
    @param catalog_path: str
    @return record array
    """
    global pyfits
    if pyfits is None:
        try:
            import astropy.io.fits as pyfits_
        except ImportError:
            try:
                import pyfits as pyfits_
            except ImportError:
                raise RuntimeError("Install astropy or pyfits to read a FITS catalog")
        pyfits = pyfits_

    return pyfits.open(catalog_path, uint=True)[1].data


def load_catalog_npy(catalog_path):
    """
    Load a numpy catalog (extension ".npy")
    @param catalog_path: str
    @return record array
    """
    return numpy.load(catalog_path)


def load_catalog_txt(catalog_path):
    """
    Load a text catalog.
    Not implemented yet. (CSV? TSV? How can I guess column names/types?)
    """

    raise NotImplementedError()


def quote_identifier(colname):
    """
    Quote colname as identifiers.
        e.g. abc"def"g -> "abc""def""g"
    @param colname: str
    @return str
    """
    return '"' + colname.replace('"', '""') + '"'


def get_format(numpy_type):
    """
    Get the % format ("%d", "%f" etc) for a given numpy type.
    @param numpy_type: type object
    @return str
    """
    return format_dict[numpy_type][0]


def get_sqltype(numpy_type):
    """
    Get the type name in SQL for a given numpy type.
        e.g. numpy.float32 -> "real"
    @param numpy_type: type object
    @return str
    """
    return format_dict[numpy_type][1]


format_dict = {
    numpy.bool_   : ("%d"   , "boolean"),
    numpy.int8    : ("%d"   , "smallint"),
    numpy.int16   : ("%d"   , "smallint"),
    numpy.int32   : ("%d"   , "integer" ),
    numpy.int64   : ("%d"   , "bigint"  ),
    numpy.uint8   : ("%d"   , "smallint"),
    numpy.uint16  : ("%d"   , "integer" ),
    numpy.uint32  : ("%d"   , "bigint"  ),
    numpy.uint64  : ("%d"   , "bigint"  ),
    numpy.float16 : ("%.4e" , "real"    ),
    numpy.float32 : ("%.8e" , "real"    ),
    numpy.float64 : ("%.16e", "double precision"),
    numpy.str_    : ("%s"   , "text"    ),
}


def load_query_template(template_path):
    """
    Load a query template.
    @param template_path: str
        file path or ':embedded:'
    @return str
    """
    if template_path == ":embedded:":
        return embeddedTemplate.strip()
    else:
        return open(template_path, "r").read().strip()

embeddedTemplate_old = """
WITH
    {user_catalog}
    ,
    match AS (
        SELECT
            object_id,
            chord_to_arcsec(coord <-> radec_to_coord({ra}, {dec})) AS match_distance,
            user_catalog.*
        FROM
            user_catalog JOIN {rerun}.forced
                ON coneSearch(coord, {ra}, {dec}, {accuracy})
        OFFSET 0 /* Suppress stupid optimization */
    )
SELECT
    match.* /* NO COMMA HERE */ {columns}
FROM
    match LEFT JOIN {rerun}.forced USING(object_id)
WHERE
    isprimary
"""

embeddedTemplate = """
WITH
    {user_catalog}
    ,
    match AS (
        SELECT
            object_id,
            user_catalog.*
        FROM
            user_catalog JOIN {rerun}.forced
                ON coneSearch(coord, {ra}, {dec}, {accuracy})
        OFFSET 0 /* Suppress stupid optimization */
    )
SELECT
    match.* /* NO COMMA HERE */ {columns}
FROM
    match LEFT JOIN {rerun}.forced USING(object_id)
WHERE
    isprimary
"""

if __name__ == "__main__":
    startup()
