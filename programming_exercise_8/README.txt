Note that indicatorMat.txt is a 28 MB file, so it has not been included in this repository.
To generate indicatorMat.txt, start Octave (or Matlab) and change the working directory to the directory that contains ex8_movies.mat.  Run the following commands:

load('ex8_movies.mat');
fid = fopen('indicatorMat.txt','w+');
for movie_idx=1:size(R,1)
    for user_idx=1:(size(R,2)-1)
        fprintf(fid,'%.15f,',R(movie_idx,user_idx));
	end
	fprintf(fid,'%.15f',R(movie_idx,size(R,2)));
	fprintf(fid, '\n');
end
fclose(fid);

Note that ratingsMat.txt is a 28 MB file, so it has not been included in this repository.
To generate ratingsMat.txt, start Octave (or Matlab) and change the working directory to the directory that contains ex8_movies.mat.  Run the following commands:

load('ex8_movies.mat');
fid = fopen('ratingsMat.txt','w+');
for movie_idx=1:size(Y,1)
    for user_idx=1:(size(Y,2)-1)
        fprintf(fid,'%.15f,',Y(movie_idx,user_idx));
	end
	fprintf(fid,'%.15f',Y(movie_idx,size(Y,2)));
	fprintf(fid, '\n');
end
fclose(fid);
