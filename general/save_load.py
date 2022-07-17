import sys
import os
import json
import pickle

BASIC_DATA_TYPE = (int, str, float)
BASIC_DATA_TYPE_BOOL = (bool)

TYPE_BASIC = "type_basic"
TYPE_BOOL = "type_bool"
TYPE_OBJECT = "type_object"
TYPE_LIST = "type_list"
TYPE_DICT = "type_dict"
TYPE_UNDEFINED = "type_undefined"


class TypeCheck:
    @staticmethod
    def is_list(obj):
        return type(obj) == list and isinstance(obj, list)

    @staticmethod
    def is_dict(obj):
        return type(obj) == dict and isinstance(obj, dict)

    @staticmethod
    def is_object(obj):
        return isinstance(obj, object)

    @staticmethod
    def is_basic(obj):
        return isinstance(obj, BASIC_DATA_TYPE)

    @staticmethod
    def is_bool(obj):
        return isinstance(obj, bool)

    @staticmethod
    def get_obj_type(obj):
        if TypeCheck.is_basic(obj):
            return TYPE_BASIC
        elif TypeCheck.is_bool(obj):
            return TYPE_BOOL
        elif TypeCheck.is_list(obj):
            return TYPE_LIST
        elif TypeCheck.is_dict(obj):
            return TYPE_DICT
        elif TypeCheck.is_object(obj):
            return TYPE_OBJECT
        else:
            return TYPE_UNDEFINED


class SaveBasic:
    @staticmethod
    def save_basic(data, fn, path=None, file_type=None, called=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if data is not None and path is not None and fn is not None:
            if file_type == 'txt':
                SaveBasic.save_txt(data, path, fn, called=called)
            elif file_type == 'json':
                SaveBasic.save_json(data, path, fn, called=called)
            else:
                SaveBasic.save_obj(data, path, fn, called=called)
        else:
            SaveBasic.save_log(called, success=False)

    @staticmethod
    def save_log(called, success=False):
        if success:
            if called and len(called):
                print(str(called) + ' : saving data success')
            else:
                print('saving data success')
        else:
            if called and len(called):
                print(str(called) + ' : saving data error')
            else:
                print('saving data error')

    @staticmethod
    def save_txt(data, path, fn, called=None):
        if os.path.isdir(path):
            with open(os.path.join(path, fn), "w", encoding='UTF-8') as f:
                for line in data:
                    f.write(line)
                    f.write('\n')
            SaveBasic.save_log(called, success=True)
        else:
            SaveBasic.save_log(called)
        pass

    @staticmethod
    def save_obj(data, path, fn, called=None):
        if os.path.isdir(path):
            with open(os.path.join(path, fn), 'wb') as f:
                pickle.dump(data, f)
            SaveBasic.save_log(called, success=True)
        else:
            SaveBasic.save_log(called)

    @staticmethod
    def save_json(data, path, fn, called=None):
        if os.path.isdir(path):
            with open(os.path.join(path, fn), 'w') as f:
                json.dump(data, f, indent=2)
            SaveBasic.save_log(called, success=True)
        else:
            SaveBasic.save_log(called)


class LoadBasic:
    @staticmethod
    def load_basic(fn, path=None, file_type=None, called=None):
        if not os.path.exists(path):
            print(called + ' : path error')
            return
        if path and fn:
            if file_type == 'txt':
                data = LoadBasic.load_txt(path, fn, called=called)
            elif file_type == 'json':
                data = LoadBasic.load_json(path, fn, called=called)
            else:
                data = LoadBasic.load_obj(path, fn, called=called)
            return data
        else:
            LoadBasic.load_log(called)
            return -1


    @staticmethod
    def load_log(called, success=False):
        if success:
            if called and len(called):
                print(str(called) + ' : loading data success')
            else:
                print('loading data success')
        else:
            if called and len(called):
                print(str(called) + ' : loading data error')
            else:
                print('loading data error')

    @staticmethod
    def load_txt(path, fn, called=None):
        if os.path.isdir(path):
            with open(os.path.join(path, fn), "r", encoding='UTF-8') as f:
                data = f.readlines()
            LoadBasic.load_log(called, success=True)
            return data
        else:
            LoadBasic.load_log(called)
        return -1

    @staticmethod
    def load_obj(path, fn, called=None):
        if os.path.isfile(os.path.join(path, fn)):
            with open(os.path.join(path, fn), 'rb') as f:
                data = pickle.load(f)
            LoadBasic.load_log(called, success=True)
            return data
        else:
            LoadBasic.load_log(called)
        return -1

    @staticmethod
    def load_json(path, fn, called=None):
        if path and os.path.isdir(path):
            with open(os.path.join(path, fn), 'r') as f:
                data = json.load(f)
            LoadBasic.load_log(called, success=True)
            return data
        else:
            LoadBasic.load_log(called)
        return -1
