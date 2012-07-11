% mock waitbar to silent the process
function h=waitbar(a,b)
if nargout >0
	h=[];
end
