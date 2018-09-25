clc
close all
clear all

ds = datastore('./images','IncludeSubfolders', true,'FileExtensions', '.png','Type', 'image');

lab = getfield(ds,'Files');

A = strings([length(lab),1]);

for i=1:length(lab)
    aux = string(lab{i});
    split_aux = strsplit(aux,'/');
    name = split_aux(end);
    split_name = strsplit(name,'_');
    A(i) = split_name(1);
end

A_lab = categorical(A);

imds = setfield(ds,'Labels',A_lab);
% imds = imageDatastore(fullfile('./images', {'F', 'ME', 'MS'}), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds)

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

[trainingSet, validationSet] = splitEachLabel(imds, 0.7, 'randomize');
% Find the first instance of an image for each category
F = find(trainingSet.Labels == 'F', 1);
ME = find(trainingSet.Labels == 'ME', 1);
MS = find(trainingSet.Labels == 'MS', 1);

% figure

subplot(1,3,1);
imshow(readimage(trainingSet,F))
subplot(1,3,2);
imshow(readimage(trainingSet,ME))
subplot(1,3,3);
imshow(readimage(trainingSet,MS))

bag = bagOfFeatures(trainingSet);
img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

confMatrix = evaluate(categoryClassifier, trainingSet);
mean(diag(confMatrix));

confMatrix = evaluate(categoryClassifier, validationSet);
mean(diag(confMatrix));