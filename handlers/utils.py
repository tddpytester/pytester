import pickle
def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
import json
def read_json_file(path):
    with open(path) as f:
        datas = f.readlines()
    output = {}
    for data in datas:
        data = json.loads(data.strip())
        for k in data:
            if k not in output:
                output[k] = []
            output[k].append(data[k])
    return output
def save_json_dataset(data_dict, filename='gts.json'):
    with open(filename,"a") as f:
        for i in range(len(data_dict)):
            json.dump(data_dict[i], f)
            f.write("\n")
def save_file_txt(data_list, file_path):
    wfile = open(file_path, 'w')
    for data in data_list:
        wfile.write(data + '\n')
    wfile.close()
