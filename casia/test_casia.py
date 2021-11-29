from datetime import datetime
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from loss import SupConLoss
from torch.cuda.amp import autocast



def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist



# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
        #! TODO: For Oumvlp, change this 10 to 13
    if not each_angle:
        result = np.mean(result)
    return result




def evaluate(model, test_loader, debug=False):

    feature_list = list()
    ids_list = list()
    condition_list = list()
    angle_list = list()

    
    if debug:
        import pickle
        with open('casia_test_feats.pkl', 'rb') as f:
            obj = pickle.load(f)
        
        features = obj['features']
        labels = obj['labels']
        angle_list = obj['angle_list']
        condition_list = obj['condition_list']
        
    else:
        model.eval()
        
        with torch.no_grad():
            dataset_length = len(test_loader)
            running_loss = 0.0

            start = time.time()
            for i, data in enumerate(test_loader):
                samples = data['data']
                ids = data['pid']
                conditions = data['condition']
                angles = data['angle']
                angles = angles.tolist()
                
                # samples = samples.cuda()
                ids_cuda = ids.cuda()

                features = []
                for clips in samples:
                    
                    inputs = clips.to(device)
                    # forward
                    o,_ = model(inputs)
                    features.append(o)
                features = torch.stack(features)

                features = features.permute(1, 0, 2)

                batch, p, _ = features.shape

                features = torch.mean(features, dim=1)
                f = features.view(batch, -1)
                
                #Concat embed of all pos samples to get embed of final video 
                feature_list.append(features.contiguous().view(batch, -1).data.cpu().numpy())

                del features, ids_cuda

                ids_list += ids
                condition_list += conditions
                angle_list += angles

        features = np.concatenate(feature_list, 0)
        # features is of size (num_sequences, embedding) - One embedding for every video in test set
        # print(features.shape)
        labels = np.array(ids_list)
        # Labels of all sequences
        print(f'Evaluation Done... Time taken:{time.time() - start}')
        print(f'Scoring started')
        if debug:
            import pickle
            saved_obj = {
                'features' : features,
                'labels' : labels,
                'angle_list' : angle_list,
                'condition_list' : condition_list
            }
            with open('casia_test_feats.pkl', 'wb') as f:
                pickle.dump(saved_obj, f)
        
    view=angle_list
    angle_list = list(set(list(angle_list)))
    angle_list.sort()
    view_num = len(angle_list)
    sample_num = len(features)

    dataset = 'CASIA'


    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                    'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}

    print(angle_list, probe_seq_dict[dataset], gallery_seq_dict[dataset])

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(angle_list):
                for (v2, gallery_view) in enumerate(angle_list):
                    gseq_mask = np.isin(condition_list, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = features[gseq_mask, :]
                    gallery_y = labels[gseq_mask]

                    pseq_mask = np.isin(condition_list, probe_seq) & np.isin(view, [probe_view])
                    probe_x = features[pseq_mask, :]
                    probe_y = labels[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                            0) * 100 / dist.shape[0], 2)
                    #explaination:
                    # probes = np.reshape(probe_y, [-1, 1])
                    # sorted_galleries = gallery_y[idx[:, 0:num_rank]]  #Top 'num_rank' elements in given configuration in test set, closest to given probe
                    # match_or_not = (probes == sorted_galleries) #boolean array, whether sorted_galleries has the correct match for probe
                    # cumulative_sum = np.cumsum(match_or_not, 1) > 0 #Traverse a row. Store cumulative sum of all elements in that row till now. Gives a 1-0 matrix, since input is boolean. > 0 again makes it a boolean matrix
                    # correct_preds = np.sum(cumulative_sum,0) # A single vector of size (num_rank), eg, [2, 6, 9, 10, 11], storing number of correct rank-1 predictions as 1st element, rank-2 correct preds as 2nd element, etc.
                    # accuracy = correct_preds * 100 / dist.shape[0]    #Convert correct_preds to accuracy, out of 100
                    # acc[p,v1,v2,:] = np.round(accuracy, 2)

                    #Q: We are not considering full gallery set to compute the intermediate 'accuracy' variable in the commented code above. 
                    #A: Yes, we consider full gallery set. We take every person, but only some combinations of that person. We then take average of scores of diffferent combinations being in probe/gallery sets. A good model should still predict correctly 
        


    scores = {}

    # Print rank-1 accuracy of the best model
    # e.g.
    # ===Rank-1 (Include identical-view cases)===
    # NM: 95.405,     BG: 88.284,     CL: 72.041
    for i in range(1):
        # print('===Rank-%d (Include identical-view cases)===' % (i + 1))
        # print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        #     np.mean(acc[0, :, :, i]),
        #     np.mean(acc[1, :, :, i]),
        #     np.mean(acc[2, :, :, i])))
        scores[f'I_{i+1}_NM'] = np.mean(acc[0, :, :, i])
        scores[f'I_{i+1}_BG'] = np.mean(acc[1, :, :, i])
        scores[f'I_{i+1}_CL'] = np.mean(acc[2, :, :, i])

    # Print rank-1 accuracy of the best model，excluding identical-view cases
    # e.g.
    # ===Rank-1 (Exclude identical-view cases)===
    # NM: 94.964,     BG: 87.239,     CL: 70.355
    for i in range(1):
        # print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
        # print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        #     de_diag(acc[0, :, :, i]),
        #     de_diag(acc[1, :, :, i]),
        #     de_diag(acc[2, :, :, i])))

        scores[f'E_{i+1}_NM'] = de_diag(acc[0, :, :, i])
        scores[f'E_{i+1}_BG'] = de_diag(acc[1, :, :, i])
        scores[f'E_{i+1}_CL'] = de_diag(acc[2, :, :, i])

    # Print rank-1 accuracy of the best model (Each Angle)
    # e.g.
    # ===Rank-1 of each angle (Exclude identical-view cases)===
    # NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
    # BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
    # CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
    # np.set_printoptions(precision=2, floatmode='fixed')
    # for i in range(1):
    #     print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
    #     print('NM:', de_diag(acc[0, :, :, i], True))
    #     print('BG:', de_diag(acc[1, :, :, i], True))
    #     print('CL:', de_diag(acc[2, :, :, i], True))

    # scores['test_loss'] = total_loss if not debug else 0
    return scores



if __name__ == '__main__':
    from models.r21d import R2Plus1DNet
    from models import r3d_pretrained, r21d_pretrained

    from datasets.casiab import CasiabDataset
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    print('Creating model...')
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
    args = parser.parse_args()

    print(vars(args))
    
    if args.model == 'r3d':
        model = r3d_pretrained.generate_model(model_depth=18).to(device)
    elif args.model == 'r21d':
        model = r21d_pretrained.generate_model(model_depth=18).to(device)
    else:
        model = C3D(with_classifier=True, num_classes=74).to(device)
    model.fc = nn.Linear(512, 74).to(device)
    print('Getting loader...')

    test_transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])
    
    test_dataset = CasiabDataset('/home/c3-0/datasets/casia-b/DatasetB_crops_v4/casiab_movingavg_crops_pad_5pixels/video', 
            16, '1', train=False, transforms_=test_transforms)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                num_workers=8, pin_memory=True)
    print(len(test_loader))

    if args.ckpt is not None:
        print("Loading weights...")
        model.load_state_dict(torch.load(args.ckpt), strict=True)
    
    model.eval()
    print('Evaluating....')
    scores = evaluate(model, test_loader, debug=False)
    print(scores)

