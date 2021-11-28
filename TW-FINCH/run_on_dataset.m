function Result = run_on_dataset(Dataset, tw_finch, datasets_path)
% Input
% Dataset: dataset name - one of 'BF', 'Hollywood_extended', 'FS', 'MPII_Cooking', 'YTI'
% tw_finch: true or false: pass true to run tw_finch else will run finch
% datasets_path : root folder where all datasets folders exist
% Output
% Result: struct with computed metrics on full Dataset and a detailed overview table of reslts per video

%%%%%%%%%%%%%
if nargin < 3
    datasets_path='./Action_Segmentation_Datasets/';
end



if tw_finch
    fprintf('Running: %s on %s Dataset\n', 'TW-FINCH', Dataset)
else
    fprintf('Running: %s on %s Dataset\n', 'FINCH', Dataset)
end

 
feat_path=fullfile(datasets_path, Dataset, 'features');

gt_path=fullfile(datasets_path, Dataset, 'groundTruth');
mapping_path = fullfile(datasets_path, Dataset, 'mapping');

imds = imageDatastore(feat_path, 'IncludeSubfolders',true,'FileExtensions','.txt','LabelSource','foldernames');


files=imds.Files;

activity_video_labels=imds.Labels;

if strcmp(Dataset, 'MPII_Cooking')
    act={};
        for i=1:numel(files)
            buff = textscan(files{i},'%s','Delimiter','/');
            buff=buff{1}; 
            vid_name = buff{end}; vid_name=vid_name(1:end-4);
        n =textscan(vid_name, '%s', 'Delimiter', '-');
        act = [act; n{1}{2}];
        end
        activity_video_labels=categorical(act);
end
%%% load MPI test set Only %%%
MPI_flag = true;
 if strcmp(Dataset, 'MPII_Cooking') && MPI_flag
    [files, activity_video_labels] = get_MPI_testset(files, activity_video_labels, mapping_path);
 end
 
 
% Load average gt per activity for clustering each video 
activity_gt = load(fullfile(mapping_path, 'avg_gt_per_activity.mat'));
gt_per_activity = activity_gt.gt_per_activity; 
 
 
[moF,iou,f1, overview_table] = action_seg(files, activity_video_labels, gt_per_activity, Dataset, datasets_path, tw_finch);
 
Result.moF = moF;
Result.IoU = iou;
Result.F1 = f1;
Result.overview_table = overview_table;


fprintf('[INFO] : Results\n')
fprintf('DataSet: %s .... MoF:= %2.1f%% .... IoU:= %2.1f%%\n', Dataset, moF*100, iou*100 )

%% get MPI_ test set
    function [files, activity_labels] = get_MPI_testset(filenames, labs, mapping_path)        
        MPI_test_path=fullfile(mapping_path, 'sequencesTest.txt');
        tb= table2cell(readtable(MPI_test_path, 'ReadVariableNames', false));
        idx=[];
        for i=1:numel(tb);
            idx= [idx; find(contains(filenames, tb{i}))];
        end
        files = filenames(idx);
        activity_labels=labs(idx);       
    end
    
end
