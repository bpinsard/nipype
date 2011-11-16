function write_sica_comps(sica, sica_comp_filename_fmt)
sica.header.dt=[16 0];
for i=1:sica.nbcomp
	d=sica.header;
	d.fname = ['./' sprintf(sica_comp_filename_fmt, i)];

	if length(size(sica.S))<3
		[vol] = st_1Dto3D(sica.S(:,i),sica.mask);
	else
		vol = squeeze(sica.S(:,:,:,i));
	end
	[vol_c] = st_correct_vol(vol,sica.mask);
	st_write_analyze(vol_c,d,d.fname);
	out.files{i}=d.fname;
end
