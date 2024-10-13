import os
import tqdm

import torch
import numpy as np
import cv2
import math
import numpy as np

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.visualization import draw_image, draw_bev_rect, draw_bev, init_bev

def alpha2rot(alpha, z3d, x3d):
    ry3d = alpha + math.atan2(-z3d, x3d) + 0.5 * math.pi

    while ry3d > math.pi: ry3d -= math.pi * 2
    while ry3d < (-math.pi): ry3d += math.pi * 2

    return ry3d

def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 9 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, 0]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h, -h/2]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2, 0]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners) 
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)


def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 4 x 4
    # return: n x 3
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d[:, :2] = pts_2d[:, :2] / pts_2d[:, 2:3]
    return pts_2d[:, :3]



class Tester(object):
    def __init__(self, cfg, model, data_loader, logger):
        self.cfg = cfg
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.cfg.get('resume_model', None):
            load_checkpoint(model = self.model,
                        optimizer = None,
                        filename = cfg['resume_model'],
                        logger = self.logger,
                        map_location=self.device)

        self.model.to(self.device)


    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.data_loader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.data_loader):
            # load evaluation data and move data to current device.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)

            calib = calibs.detach().cpu().numpy().squeeze().copy()
             

  

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image_2d = inputs.detach().cpu().numpy().squeeze().copy()
            image_2d = image_2d.transpose((1, 2, 0))
            image_2d = (image_2d * std + mean) * 255
            image_2d = image_2d.astype(np.uint8)
            image_2d = cv2.cvtColor(image_2d, cv2.COLOR_RGB2BGR)
             
            # the outputs of centernet
            outputs = self.model(inputs,coord_ranges,calibs,K=50,mode='test')
            dets = extract_dets_from_outputs(outputs=outputs, K=50)
            dets = dets.detach().cpu().numpy()
            # image_2d = inputs.copy()
            
 
            image_bev = init_bev()


            # get corresponding calibs & transform tensor to numpy
            calibs = [self.data_loader.dataset.get_calib(index)  for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.data_loader.dataset.cls_mean_size
            dets = decode_detections(dets = dets,
                                     info = info,
                                     calibs = calibs,
                                     cls_mean_size=cls_mean_size,
                                     threshold = self.cfg['threshold'])
            
            # draw_bev(image_bev, x3d, z3d, l3d, w3d, ry3d)
            for img_id in dets.keys():
                for i in range(len(dets[img_id])):
                    dets[img_id][i]
                    # print(i, dets[img_id][i])
                    h3d = dets[img_id][i][6]
                    w3d = dets[img_id][i][7]
                    l3d = dets[img_id][i][8]

                    alpha = dets[img_id][i][1]

                    # ry3d = dets[img_id][i][12]
                    

                    x3d = dets[img_id][i][9]
                    y3d = dets[img_id][i][10]
                    z3d = dets[img_id][i][11]

                    ry3d = alpha2rot(alpha, z3d, x3d)
                    # ry3d = dets[img_id][i][12]
                     
                    point_3d = compute_box_3d([h3d, w3d, l3d], [x3d, y3d, z3d], ry3d)
                    point_2d = project_to_image(point_3d, calib)
                    draw_image(image_2d, point_2d)
                    draw_bev(image_bev, x3d, z3d, l3d, w3d, ry3d)
                    
            image_bev = cv2.resize(image_bev, (image_2d.shape[0], image_2d.shape[0]))
            image_visual = cv2.hconcat([image_2d, image_bev])
             
            cv2.imshow("dets", image_visual)
            cv2.waitKey(0)
                 
            # for i in len(dets):
                # det = dets[i]
                # print(det)
            results.update(dets)
            progress_bar.update()

        # save the result for evaluation.
        self.save_results(results)
        progress_bar.close()

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)
        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()







