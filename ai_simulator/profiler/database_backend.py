import os
from abc import ABC
import json


class DatabaseBackendException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def print_error_info(self):
        print(self.error_info)


class DatabaseBackend(ABC):
    def __init__(self):
        pass

    def get(self, key):
        pass

    def put(self, key, value):
        pass

    def __del__(self):
        pass


class DatabaseBackendLocalFile(DatabaseBackend):
    '''
    Basic database class, save profiling result in files.
    Load all files of given path in memory when created.
    '''

    def __init__(self, db_file_path):
        '''
        Start a file based default database backend.

        Args:
            db_file_path:    The path of database file.
        '''
        # Original database content.
        self.__db_file_path = db_file_path
        self.__database = {}
        db_file_dir = os.path.dirname(db_file_path)
        if not os.path.exists(db_file_dir):
            os.makedirs(db_file_dir)
        self.__loadDatabaseFile(db_file_path)
        return

    def __del__(self):
        '''
        Finalize the database backend. Save all new records into the
        new_record_file.
        '''
        with open(self.__db_file_path, 'w') as fd_out:
            json.dump(self.__database, fd_out, indent=4)

    def __loadDatabaseFile(self, db_file_path):
        '''
        Load all database files in given dir.
        All .pfdb files in the given path are regarded as part of the DB.
        First save a key, and the following line saves its profile result.
        Then repeat. Return number of files loaded
        '''
        with open(self.__db_file_path, 'r') as fd_in:
            self.__database = json.load(fd_in)

    def put(self, key, result):
        self.__database[key] = result

    def get(self, key):
        if key in self.__database:
            return self.__database[key]
        return None
