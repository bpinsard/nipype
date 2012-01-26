function write_sica_comps(sica, sica_fname)

ni = nifti;
ni.dat = file_array(sica_fname, ...
										[sica.header.dim sica.nbcomp], ...
										[16 0], ...
										0,1,0);
ni.mat = sica.header.mat;
ni.mat0= sica.header.mat;
ni.descrip = 'ICA components';
create(ni);
data=zeros([sica.header.dim sica.nbcomp]);
for i=1:sica.nbcomp
	data(:,:,:,i) = st_1Dto3D(sica.S(:,i),sica.mask);
end
%data = st_correct_vol(data,sica.mask);
for i=1:sica.nbcomp;
	ni.dat(:,:,:,i) = data(:,:,:,i);
end