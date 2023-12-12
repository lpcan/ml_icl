#!/bin/env python3
import json
import argparse
import urllib.request, urllib.error, urllib.parse
import time
import sys
import csv
import getpass
import os
import os.path
import re
import ssl

from astropy.table import Table
import io
import warnings

version = 20190514.1

args = None

def sql_query(query, user, release_version):
    credential = {'account_name': user, 'password': getPassword()}
    sql = query

    job = None

    try:
        job = submitJob(credential, sql, 'fits', release_version)
        blockUntilJobFinishes(credential, job['id'])
        res = download(credential, job['id'], sys.stdout.buffer)
        res = parse_query_result(res)
        print(res)
        deleteJob(credential, job['id'])
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print('invalid id or password.', file=sys.stderr)
        if e.code == 406:
            print(e.read(), file=sys.stderr)
        else:
            print(e, file=sys.stderr)
    except QueryError as e:
        print(e, file=sys.stderr)
    except KeyboardInterrupt:
        if job is not None:
            jobCancel(credential, job['id'])
        raise
    else:
        return res

    sys.exit(1) 

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--user', '-u', required=True,
                        help='specify your account name')
    parser.add_argument('--release-version', '-r', choices='pdr1 pdr2 pdr2-citus pdr3 pdr3-citus pdr3-citus-columnar'.split(), default='pdr3',
                        help='specify release version')
    parser.add_argument('--delete-job', '-D', action='store_true',
                        help='delete the job you submitted after your downloading')
    parser.add_argument('--format', '-f', dest='out_format', default='fits', choices=['csv', 'csv.gz', 'sqlite3', 'fits'],
                        help='specify output format')
    parser.add_argument('--nomail', '-M', action='store_true',
                        help='suppress email notice')
    parser.add_argument('--password-env', default='HSC_SSP_CAS_PASSWORD',
                        help='specify the environment variable that has password as its content')
    parser.add_argument('--preview', '-p', action='store_true',
                        help='quick mode (short timeout)')
    parser.add_argument('--skip-syntax-check', '-S', action='store_true',
                        help='skip syntax check (Use if you get 502: Proxy Error)')
    parser.add_argument('--api-url', default='https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/',
                        help='for developers')
    parser.add_argument('sql-file', type=argparse.FileType('r'),
                        help='SQL file')

    global args
    args = parser.parse_args()

    credential = {'account_name': args.user, 'password': getPassword()}
    sql = args.__dict__['sql-file'].read()

    job = None

    try:
        if args.preview:
            preview(credential, sql, sys.stdout)
        else:
            job = submitJob(credential, sql, args.out_format, args.release_version)
            blockUntilJobFinishes(credential, job['id'])
            res = download(credential, job['id'], sys.stdout.buffer)
            res = parse_query_result(res)
            print(res)
            if args.delete_job:
                deleteJob(credential, job['id'])
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print('invalid id or password.', file=sys.stderr)
        if e.code == 406:
            print(e.read(), file=sys.stderr)
        else:
            print(e, file=sys.stderr)
    except QueryError as e:
        print(e, file=sys.stderr)
    except KeyboardInterrupt:
        if job is not None:
            jobCancel(credential, job['id'])
        raise
    else:
        sys.exit(0)

    sys.exit(1)


class QueryError(Exception):
    pass


def httpJsonPost(url, data):
    data['clientVersion'] = version
    postData = json.dumps(data)
    return httpPost(url, postData, {'Content-type': 'application/json'})


def httpPost(url, postData, headers):
    req = urllib.request.Request(url, postData.encode('utf-8'), headers)
    res = urllib.request.urlopen(req)
    return res


def submitJob(credential, sql, out_format, release_version):
    url = 'https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/' + 'submit'
    catalog_job = {
        'sql'                     : sql,
        'out_format'              : out_format,
        'include_metainfo_to_body': True,
        'release_version'         : release_version,
    }
    postData = {'credential': credential, 'catalog_job': catalog_job, 'nomail': True, 'skip_syntax_check': True}
    res = httpJsonPost(url, postData)
    job = json.load(res)
    return job


def jobStatus(credential, job_id):
    url = 'https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/' + 'status'
    postData = {'credential': credential, 'id': job_id}
    res = httpJsonPost(url, postData)
    job = json.load(res)
    return job


def jobCancel(credential, job_id):
    url = 'https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/' + 'cancel'
    postData = {'credential': credential, 'id': job_id}
    httpJsonPost(url, postData)


def preview(credential, sql, out):
    url = 'https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/' + 'preview'
    catalog_job = {
        'sql'             : sql,
        'release_version' : args.release_version,
    }
    postData = {'credential': credential, 'catalog_job': catalog_job}
    res = httpJsonPost(url, postData)
    result = json.load(res)

    writer = csv.writer(out)
    # writer.writerow(result['result']['fields'])
    for row in result['result']['rows']:
        writer.writerow(row)

    if result['result']['count'] > len(result['result']['rows']):
        raise QueryError('only top %d records are displayed !' % len(result['result']['rows']))


def blockUntilJobFinishes(credential, job_id):
    max_interval = 5 * 60 # sec.
    interval = 1
    while True:
        time.sleep(interval)
        job = jobStatus(credential, job_id)
        if job['status'] == 'error':
            raise QueryError('query error: ' + job['error'])
        if job['status'] == 'done':
            break
        interval *= 2
        if interval > max_interval:
            interval = max_interval


def download(credential, job_id, out):
    url = 'https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/' + 'download'
    postData = {'credential': credential, 'id': job_id}
    res = httpJsonPost(url, postData)
    # bufSize = 64 * 1<<10 # 64k
    # while True:
    #     buf = res.read(bufSize)
    #     out.write(buf)
    #     if len(buf) < bufSize:
    #         break
    return res # from unagi

def parse_query_result(response, verbose=False):
    """
    Parse the SQL result to something readable.
    """
    try:
        # Convert the output into Astropy.table
        result = Table.read(
            io.BytesIO(response.read(response.length)))
        # Remove _isnull columns
        columns = [col for col in result.colnames if not col.endswith('_isnull')]
        result = result[columns]
    except Exception as e:
        print(e)
        print("\n# Cannot convert search result into Astropy table.")

    if not result and verbose:
        warnings.warn('Query returned no results, so the table will be empty!')

    return result

def deleteJob(credential, job_id):
    url = 'https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs/' + 'delete'
    postData = {'credential': credential, 'id': job_id}
    httpJsonPost(url, postData)


def getPassword():
    if args is not None:
        pass_env = args.password_env
    else:
        pass_env = 'HSC_SSP_CAS_PASSWORD'
    password_from_envvar = os.environ.get(pass_env, '')
    if password_from_envvar != '':
        return password_from_envvar
    else:
        return getpass.getpass('password? ')


if __name__ == '__main__':
    main()
