function [mof, iou, fscore, res] = evaluate(label_pre, label_gt)
% compute Hungarian based accuracy and IOU score
res = bestMap(label_gt(:), label_pre(:));
mof = length(find(label_gt(:) == res(:)))/length(label_gt(:));

k = length(unique(label_pre));
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

end