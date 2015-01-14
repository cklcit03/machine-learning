Note that spamTrain.txt is a 133 MB file, so it has not been included in this repository.
To generate spamTrain.txt, start Octave (or Matlab) and change the working directory to the directory that contains spamTrain.mat.  Run the following commands:

load('spamTrain.mat');
fid = fopen('spamTrain.txt','w+');
for ex_idx=1:size(X,1)
	for feature_idx=1:size(X,2)
		fprintf(fid,'%.15f,',X(ex_idx,feature_idx));
	end
	fprintf(fid,'%d',y(ex_idx));
    fprintf(fid, '\n');
end
fclose(fid);

Note that spamTest.txt is a 33 MB file, so it has not been included in this repository.
To generate spamTest.txt, start Octave (or Matlab) and change the working directory to the directory that contains spamTest.mat.  Run the following commands:

load('spamTest.mat');
fid = fopen('spamTest.txt','w+');
for ex_idx=1:size(Xtest,1)
	for feature_idx=1:size(Xtest,2)
		fprintf(fid,'%.15f,',Xtest(ex_idx,feature_idx));
	end
	fprintf(fid,'%d',ytest(ex_idx));
    fprintf(fid, '\n');
end
fclose(fid);
