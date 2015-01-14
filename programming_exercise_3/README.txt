Note that digitData.txt is a 35 MB file, so it has not been included in this repository.
To generate digitData.txt, start Octave (or Matlab) and change the working directory to the directory that contains ex3data1.mat.  Run the following commands:

load('ex3data1.mat');
write arrays X, y to CSV file
fid = fopen('digitData.txt','w+');
for ex_idx=1:size(X,1)
	for feature_idx=1:size(X,2)
		fprintf(fid,'%.15f,',X(ex_idx,feature_idx));
	end
	fprintf(fid,'%d',y(ex_idx));
    fprintf(fid, '\n');
end
fclose(fid);
