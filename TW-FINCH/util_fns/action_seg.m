
function [moF, iou, f1, overview_table] = action_seg(files, activity_video_labels, gt_per_activity, Dataset, data_root, tw_finch)

     
verbose=false;
dist = 'cosine'; 
%tw_finch = true;


each_vid_acc=[];
gt_clusters = [];
activity_labels=[];
each_vid_iou=[];
each_vid_fscore=[];

for video=1:numel(files)
       [vid_feats, gt_label_frame, vid_name]= read_video(files, data_root, video, Dataset);

        gt = gt_per_activity.(string(activity_video_labels(video)));% average gt based on per activity     
        
        %% cluster
        [c, num_clust]=FINCH(vid_feats,[], tw_finch, verbose, dist);             
        % get required clusters at given k
        ind = find(num_clust>=gt);
        ind = ind(end);
        f_c= req_numclust(c(:, ind), vid_feats, gt, tw_finch, dist);
                      
        % evaluate
        [acc, iou_score, fscore, predicted_label] = evaluate(f_c, gt_label_frame);
                 
         each_vid_acc = [each_vid_acc; acc];
         each_vid_iou = [each_vid_iou; iou_score];         
         each_vid_fscore = [each_vid_fscore; fscore];  
         
         video_name{video, 1} = vid_name;
        
        num_clust_all{video, 1} = num_clust;
        gt_clusters = [gt_clusters; gt];
        activity_labels = [activity_labels; activity_video_labels(video)];   
      
      if mod(video, 100)==0 
        fprintf('video processed: %d  done....\n', video)
      end
   end
    
     overview_table= table(video_name, ...
         each_vid_acc, each_vid_iou, each_vid_fscore, num_clust_all, gt_clusters, activity_labels);
     moF = mean(each_vid_acc);
     iou = mean(each_vid_iou);
     f1  = mean(each_vid_fscore);
     
end