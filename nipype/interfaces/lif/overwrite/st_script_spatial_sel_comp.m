function compSelInfo = st_script_spatial_sel_comp(sica_file,mask_file,opt)

% compSelInfo = st_script_spatial_sel_comp(sica_file,mask_file,opt)
%
% INPUTS
% sica_file: (string) the full path of a .mat file with a sica structure
%             saved in it OR a sica structure.
%             WARNING: the name of the files used to compute sica will be
%             used again here to access the data.
%
% mask_file: (string) the full path name of such mask in analyze format
%
% opt:  (structure) opt.nb_clust: (default floor(nbvox/10), where nbvox is
%                               the number of voxels in the region)
%                   opt.p : (default 0.0001) the p-value of the stepwise
%                           regression
%                   opt.nb_kmeans : (default 3) the number of repetition
%                   for kmeans clustering.
%                   opt.type_score: (default 'freq') type of computed score. 'freq' for the frequency of
%                   selection of the regressor and 'inertia' for the relative part of
%                   inertia explained by the clusters "selecting" the
%                   regressor
%                   opt.scoreThres : (default -1) the threshold of
%                   the scores to select the components. value between 0
%                   and 1. =-1 for automatic threshold by Otsu algorithm.
%
% OUTPUTS
% compSelInfo: a compSelInfo structure.
%
% COMMENTS
% First coded, PB 2006/05/04.

% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.

if nargin < 3
    opt = [];
end
%%% Default parameters %%%
if isfield(opt,'type_score')
    type_score = opt.type_score;
else
    type_score = 'freq';
end

if isfield(opt,'nb_clust')
    NC = opt.nb_clust;
else
    NC = Inf;
end

if isfield(opt,'nb_kmeans')
    NR = opt.nb_kmeans;
else
    NR = 3;
end

if isfield(opt,'p')
    p = opt.p;
else
    p = 0.0001;
end

if isfield(opt,'scoreThres')
    sThres = opt.scoreThres;
else
    sThres = -1;
end


%%%%% Loading the sica structure %%%%%%
if ~ischar(sica_file);
    fprintf('Loading and pre-processing data')
    sica = sica_file;
    clear sica_file
else
    fprintf('Loading and pre-processing data %s',sica_file);
    tmp = load(sica_file);
    list_fields = fieldnames(tmp);
    sica = getfield(tmp,list_fields{1});
    clear tmp
end

%%%%% Loading the mask of noise regions %%%%%
masks={};
for mf=1:size(mask_file,1)
    mask_file{mf}
    masks{mf} = st_read_vol(mask_file{mf},[],0);
end
maskLoad=cat(4,masks{:});
Siz = size(maskLoad);
maskOI = sica.mask>0;
if size(maskLoad,4)>1
    mask = zeros(Siz(1:3));
    for pp=1:size(maskLoad,4)
        mask(squeeze(maskLoad(:,:,:,pp))>0 & maskOI>0)=pp;
    end
else
    mask = uint16(round(maskLoad));
end

%%%%% Loading and pre-processing data in noise regions %%%%%
list_files = sica.data_name;
nb_rois = length(unique(mask))-1;

if (exist(list_files{1}(1,:)))&(strcmp(list_files{1}(1,end-2:end),'img'))
    for num_roi = 1:nb_rois
        sigs{num_roi} = [];
    end

    for num_run = 1:length(list_files)
        courbes_N = st_read_analyze_tc(mask,list_files{num_run},'',0,0);
        for num_roi = 1:nb_rois
            courbes_N{num_roi} = courbes_N{num_roi}(sica.suppress_vol+1:end,:);
            if isfield(sica,'detrend')
                if sica.detrend ~= -1
                    courbes_N{num_roi} = st_detrend_array(courbes_N{num_roi},sica.detrend);
                end
            end
            if isfield(sica,'slice_correction')
                if sica.slice_correction ==1
                    courbes_N{num_roi} = st_normalise(courbes_N{num_roi},2);
                    courbes_N{num_roi} = st_correct_slice_intensity(courbes_N{num_roi},mask==num_roi);
                end
            end
            if isfield(sica,'filter')
                if isfield(sica.filter,'lp')
                    if sica.filter.lp > 0
                        courbes_N{num_roi} = st_filter_data(courbes_N{num_roi},mask==num_roi,sica.TR,sica.filter.lp,'lp');
                    end
                end
                if isfield(sica.filter,'hp')
                    if sica.filter.lp > 0
                        courbes_N{num_roi} = st_filter_data(courbes_N{num_roi},mask==num_roi,sica.TR,sica.filter.hp,'hp');
                    end
                end                
            end
            courbes_N{num_roi} = st_normalise(courbes_N{num_roi},0);
            sigs{num_roi} = [sigs{num_roi} ; courbes_N{num_roi}];
        end
    end

    for num_roi = 1:nb_rois
        sigs{num_roi} = st_normalise(sigs{num_roi},0);
    end

elseif (exist(list_files{1}(1,:)))&(strcmp(list_files{1}(1,end-2:end),'mnc'))

    for num_roi = 1:nb_rois
        sigs{num_roi} = [];
    end

    for num_run = 1:length(list_files)
        data = st_read_vol(list_files{num_run},[],0);
        [nx,ny,nz,nt] = size(data);
        data = reshape(data,[nx*ny*nz nt]);        
        for num_roi = 1:nb_rois
            courbes_N{num_roi} = data(mask==num_roi,:)';
            courbes_N{num_roi} = st_detrend_array(courbes_N{num_roi},sica.detrend);
            if isfield(sica,'slice_correction')
                if sica.slice_correction ==1
                    courbes_N{num_roi} = st_normalise(courbes_N{num_roi},2);
                    courbes_N{num_roi} = st_correct_slice_intensity(courbes_N{num_roi},mask==num_roi);
                end
            end
            if isfield(sica,'filter')
                if sica.filter > 0
                    courbes_N{num_roi} = st_filter_data(courbes_N{num_roi},mask==num_roi,sica.TR,sica.filter);
                end
            end
            courbes_N{num_roi} = st_normalise(courbes_N{num_roi},0);
            sigs{num_roi} = [sigs{num_roi} ; courbes_N{num_roi}];
        end
    end

    for num_roi = 1:nb_rois
        sigs{num_roi} = st_normalise(sigs{num_roi},0);
    end
    
else
    data = (sica.A)*(sica.S');
    mask_roi = mask(sica.mask>0);
    for num_roi = 1:nb_rois
        sigs{num_roi} = st_normalise(data(:,mask_roi == num_roi),0);
    end
end

%%%%%%%% Stepwise regression %%%%%%%%%
fprintf('Performing stepwise regression \n')
reg = st_normalise(sica.A,0);
clear sica
nbvox = size(sigs{1},2);
nbclasses = min(NC,floor(nbvox/10));
if ~(nbclasses == NC)&(NC~=Inf)
    warndlg(strcat('corrected number of clusters =',num2str(nbclasses)),'kmeans');
end
[intersec,selecVector,selecInfo] = st_automatic_selection(sigs,reg,p,NR,nbclasses,type_score,0);
[A,B] = sort(selecVector);
sortS = A(end:-1:1);
num_comp = B(end:-1:1);

if sThres == -1
    %%%%%%%%%% suggested threshold using Otsu algorithm %%%%%%%%%%%%
    [H,X] = hist(selecVector,floor(length(selecVector)/5));
    H = H/sum(H);
    ngr = length(H);
    somme = sum((1:(ngr)).*H);
    eps = 10^(-10);
    seuil = 0;
    smax = 0;
    p = 0;
    a = 0;
    for i = 1:(ngr-1)
        a = a+(i)*H(i);
        p = p+H(i);
        s = somme*p-a;
        d = p*(1-p);
        if (d>=eps)
            s = s*s/d;
            if (s >= smax)
                smax = s;
                amax = a;
                pmax = p;
                seuil = i;
            end
        end
    end
    seuil = seuil-1;

    if seuil > 0
        sugThres = X(seuil);
    else
        sugThres = mean(selecVector);
    end
    sugThres = round(1000*sugThres)/1000;

else
    %%%%%%%%%% fixed threshold %%%%%%%%%%%%
    sugThres = sThres;
end

I=find(sortS>sugThres);

% Building output structure

compSelInfo.numcomp = num_comp(I);
compSelInfo.score = selecVector;
compSelInfo.thres = sugThres;
compSelInfo.add = [];
compSelInfo.rem = [];
% compSelInfo.priortype = 'spatial';
% compSelInfo.index = type_score;
% compSelInfo.mask = mask;
% compSelInfo.model = [];
% compSelInfo.win = [];
% compSelInfo.date = datestr(now);
% %compSelInfo.name = strcat(compSelInfo.priortype,'_',get(handles.listbox_sica,'string'),'_',compSelInfo.date);
% compSelInfo.name = strcat(compSelInfo.priortype,'_',compSelInfo.date);
% compSelInfo.label = 'N/A';


function data_d = st_detrend_array(data,pow)

% On effectue la regression
nt = size(data,1);
for i = 0:pow
    X(:,i+1) = ((1:nt)').^i;
end
% - calcul des betas
beta = (pinv(X'*X)*X')*data;

% - calcul des residus
data_d = data - X*beta;
