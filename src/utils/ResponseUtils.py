import random
import json

from src.configuration.config import legac_ohif_text


def load_json(file_name):
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
    return data

def getOHIFObjects(sop_instance_uid, study_instance_uid, series_instance_uid, seg_info):
    legacy_json_List = []
    sample_response = load_json("resources/AI_Response.json")
    sample_response['studyInstanceUID'] = study_instance_uid
    sample_response['seriesInstanceUID'] = series_instance_uid
    sample_response['sopInstanceUID'] = sop_instance_uid
    sample_response['segInfo'] = seg_info
    for x in seg_info:
        sample_legacy_response = load_json("resources/AI_res.json")
        sample_legacy_response['studyInstanceUID'] = study_instance_uid
        sample_legacy_response['seriesInstanceUID'] = series_instance_uid
        sample_legacy_response['sopInstanceUID'] = sop_instance_uid
        # sample_legacy_response['measurementData']['text'] = legac_ohif_text[x["seg_class"]]
        sample_legacy_response['measurementData']['text'] = seg_info
        # sample_legacy_response['measurementData']['text'] = "fixed"
        legacy_json_List.append(sample_legacy_response)
    sample_response['legacy_ohif_json'] = legacy_json_List
    return sample_response


def random_true():
    return random.random() < 0.1

# data = load_json('/Users/radpretation/PycharmProjects/rp-ai-ich-model/resources/AI_res.json')
# print("hi")