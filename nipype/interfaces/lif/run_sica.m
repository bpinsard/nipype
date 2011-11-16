sica = st_script_sica({input_file},opt_sica);
%load([job.output_dir{1} filesep 'sica.mat']);
save([job.output_dir{1} filesep 'sica.mat'], 'sica');
out.sica_mat{1}=[job.output_dir{1} filesep 'sica.mat'];

sica.header.dt=[16 0];
for i=1:sica.nbcomp
	d=sica.header;
	d.fname = [job.output_dir{1} filesep sprintf('sica_comp%04d.img', i)];

	if length(size(sica.S))<3
		[vol] = st_1Dto3D(sica.S(:,i),sica.mask);
	else
		vol = squeeze(sica.S(:,:,:,i));
	end
	[vol_c] = st_correct_vol(vol,sica.mask);
	st_write_analyze(vol_c,d,d.fname);
	out.files{i}=d.fname;
end

d=sica.header;
d.fname = [job.output_dir{1} filesep 'maskB.img'];
st_write_analyze(double(sica.mask),d,d.fname);
out.mask{1} = d.fname;
clear sica
