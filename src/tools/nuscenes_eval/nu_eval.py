import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
# import torch
# import torch.utils.data as data
import json
_valid_ids  = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def _to_float(x):
    return float("{:.2f}".format(x))

def convert_eval_format(all_bboxes):
    detections = []
    for image_id in all_bboxes:
        if type(all_bboxes[image_id]) != type({}):
            # newest format
            for j in range(len(all_bboxes[image_id])):
                item = all_bboxes[image_id][j]
                cat_id = item['class'] - 1
                category_id = _valid_ids[cat_id]
                bbox = item['bbox']
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                bbox_out = list(map(_to_float, bbox[0:4]))
                detection = {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": bbox_out,
                    "score": float("{:.2f}".format(item['score']))
                }
                detections.append(detection)
    return detections

if __name__ == '__main__':
    path = '/home/zhuoyu/Documents/inciepo/CenterTrack/exp_results_data/'
    coco = coco.COCO(path + "val.json")
    with open(path + 'nu_3d_det/save_results_nuscenes.json' ) as f:
        result = json.load(f)
    det = convert_eval_format(result)
    json.dump(det,
              open('{}nu_3d_det/results_coco.json'.format(path), 'w'))
    coco_dets = coco.loadRes(path + 'nu_3d_det/results_coco.json')
    coco_eval = COCOeval(coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    a =1
    b =2
