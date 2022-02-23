import hashlib
from abc import ABCMeta, abstractmethod
from pathlib import Path

from datadings.index import (load_keys, load_offsets, write_filter,
                             write_key_hashes, write_keys, write_offsets)
from datadings.tools import hash_md5hex, make_printer, path_append, query_user
from datadings.tools.msgpack import make_packer
from datadings.writer import Writer


class CustomWriter(Writer):
    def __init__(self, outfile, buffering=0, overwrite=False, **kwargs):
        outfile = Path(outfile)
        self._path = outfile
        outfile.parent.mkdir(parents=True, exist_ok=True)
        if outfile.exists() and not overwrite:
            self._keys = load_keys(outfile)
            self._keys_set = set()
            for key in self._keys:
                self._keys_set.add(key)
            self._offsets = list(load_offsets(outfile))
            self.written = len(self._keys)
            self._outfile = outfile.open('ab', buffering)
        else:
            self._outfile = outfile.open('wb', buffering)
            self._keys = []
            self._keys_set = set()
            self._offsets = [0]
            self.written = 0
        self._hash = hashlib.md5()
        self._packer = make_packer()
        if 'desc' not in kwargs:
            kwargs['desc'] = outfile.name
        self._disable = kwargs.get('disable', False)

    def _write_data(self, key, packed):
        if key in self._keys_set:
            raise ValueError('duplicate key %r not allowed' % key)
        self._keys.append(key)
        self._keys_set.add(key)
        self._hash.update(packed)
        self._outfile.write(packed)
        self._offsets.append(self._outfile.tell())
        self.written += 1

    def close(self):
        """
        Flush and close the dataset file and write index and MD5 files.
        """
        self._outfile.flush()
        self._outfile.close()
        paths = [
            write_offsets(self._offsets, self._path),
            write_keys(self._keys, self._path),
            write_key_hashes(self._keys, self._path),
            write_filter(self._keys, self._path),
        ]
        with path_append(self._path, '.md5').open('w', encoding='utf-8') as f:
            f.write(f'{self._hash.hexdigest()}  {self._path.name}\n')
            for path in paths:
                f.write(f'{hash_md5hex(path)}  {path.name}\n')
        if not self._disable:
            print('%d samples written' % self.written)


class CustomFileWriter(CustomWriter):
    """
    Writer for file-based datasets.
    Requires sample dicts with a unique ``"key"`` value.
    """

    def write(self, sample):
        self._write(sample['key'], sample)
