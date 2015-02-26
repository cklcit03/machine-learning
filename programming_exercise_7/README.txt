Note that ex7faces.txt is a 96 MB file, so it has not been included in this repository.
To generate ex7faces.txt, start Octave (or Matlab) and change the working directory to the directory that contains ex7faces.mat.  Run the following commands:

load ('ex7faces.mat')
fid = fopen('ex7faces.txt','w+');
for ex_idx=1:size(X,1)
    for feature_idx=1:(size(X,2)-1)
        fprintf(fid,'%.15f,',X(ex_idx,feature_idx));
    end
    fprintf(fid,'%.15f',X(ex_idx,size(X,2)));
    fprintf(fid, '\n');
end
fclose(fid);
