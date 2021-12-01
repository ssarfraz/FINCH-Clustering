function [mof, iou, fscore, res] = eval_fs(label_pre, label_gt, datasets_path)
% Evaluates 50Salads dataset in Eval Mode

% compute Hungarian based accuracy and IOU score
res = bestMap(label_gt(:), label_pre(:));

label_gt = fs_eval_mode_map(label_gt, datasets_path);         
res = fs_eval_mode_map(res, datasets_path);



mof = length(find(label_gt(:) == res(:)))/length(label_gt(:));

k = length(unique(res));
% compute IOU 
try
  iou= jaccard(categorical(label_gt), categorical(res));
catch
  iou= jaccard(label_gt, res);
end
% penalize under/over clustering equally in iou
iou(isnan(iou))=0;
iou= sum(iou)/ k;

% compute fscore
[fscore, ~, ~]=compute_clustScores(label_gt, res);


 function labs = fs_eval_mode_map(labs, datasets_path)
        mapping_path = fullfile(datasets_path, '50Salads', 'mapping');
        map=readtable(fullfile(mapping_path, 'mapping.txt'));
        label_str = map.Var2(labs);
        
        map_val=readtable(fullfile(mapping_path, 'mappingeval.txt'));
        map2=table([1:numel(map_val.Var2)]', 'RowNames', map_val.Var2);
        mapped_label = table2array(map2(label_str,1));
        grp = map_val.Var1 +1;
        labs = grp(mapped_label);
    end
end