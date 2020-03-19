function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)

%% Determine the reaching angle before movement starts at 320ms

%spikeCOunt for a single trial, single reaching angle, all neurons 
spikeCount = zeros(98,1);

%find out if time < 320ms
if length(test_data.spikes) <= 320
    % number of spikes
    for i = 1:98
        %locate all spikes and count them 
        number_of_spikes = length(find(test_data.spikes(i,1:320)==1));
        spikeCount(i) = number_of_spikes;
    end

% usig the knn classification, find the reaching angle direction using the
% training data variables spikes and reachingAngle'
%12 seems to be     the optimum number of neighbours to compare to???
    classification = knn(modelParameters.KNN_spikes, modelParameters.KNN_reaching', spikeCount', 25);
    %KNN classifys into 1x9 vector, first column is irrelevant so subtract
    %one and set that as the direcrion where direction is a number 1-8 
    directionx = find(classification == 1)-1;
    %directionx = randi(8);
else
    % if the hand has started moving don't reupdate direction, just use old
    % direction
    directionx = modelParameters.direction;
end

%% predict movement
checkx = [];
checky = [];

% current time window, get data in 20ms bites 
tmin = length(test_data.spikes)-20;
tmax = length(test_data.spikes);

% find firing rate of test data
firingRate = zeros(98,1);
for i = 1:98
    %form firing rate of test data 
    spike_location = find(test_data.spikes(i,tmin:tmax)==1);
    number_of_spikes = length(spike_location);
    firingRate(i) = number_of_spikes/(20*0.001);
end

% estimate position using the linear regression models calculated from the
% training data for each hand bin
checkx = modelParameters.LR(directionx).xpos;
checky = modelParameters.LR(directionx).ypos;
%extract the position for a single neuron
pos_x = firingRate'*checkx(:,1);
pos_y = firingRate'*checky(:,1);
%find velocity  
velocity_x = firingRate'*modelParameters.LR(directionx).xpos;
velocity_y = firingRate'*modelParameters.LR(directionx).ypos;

%% Find positions 
%don't update until moving starts otherwise use starthandpos
if length(test_data.spikes) <= 320
    x = test_data.startHandPos(1);
    y = test_data.startHandPos(2);
else
    %update position based on velocity
    x = test_data.decodedHandPos(1,length(test_data.decodedHandPos(1,:))) + velocity_x*(20*0.001);
    y = test_data.decodedHandPos(2,length(test_data.decodedHandPos(2,:))) + velocity_y*(20*0.001);
    
    %max trajectory check, if the trajectory is greater than the max from
    %the training data, reset it 
    if sqrt(x^2+y^2) - sqrt(modelParameters.LR(directionx).maxx^2 + modelParameters.LR(directionx).maxy^2) > 0
         x = modelParameters.LR(directionx).maxx;
         y = modelParameters.LR(directionx).maxy;
    end
    %end trajectory checker to make sure it reaches max positionn 
    if tmax > 470
        x = modelParameters.LR(directionx).maxx;
        y = modelParameters.LR(directionx).maxy;
    end
    
end
%update newmodelparamaters to ensure that the direction keeps updating 
newModelParameters.LR = modelParameters.LR;
newModelParameters.direction = directionx;
newModelParameters.KNN_spikes = modelParameters.KNN_spikes;
newModelParameters.KNN_reaching = modelParameters.KNN_reaching;

end

%% KNN model
%inputs
%training data = spike count data for training
%training_labels = hand bin labels for specfic spike count data
%tests_data = spike count for test data
% k = number of nearest neighbours 

function labels = knn(training_data, training_labels, test_data, k)

test_length = size(test_data,2);
training_length = size(training_data,1);

%training labels currently nx1m where length is in and classifcation is
%from 1-8, need to split into 9 binary channel for classification 
nClasses = max(training_labels)+1;
%restructure training labels
train_labels_new = zeros(training_length,nClasses);

    for i=1:training_length
        train_labels_new(i,training_labels(i)+1) = 1;
    end

training_labels = train_labels_new;
%find new number of classes
nClasses = size(training_labels,2);

nTestPoints = size(test_data,1);

labels = zeros(nTestPoints, nClasses);

for i=1:nTestPoints
  % use Euclidean distance to find the difference between the two data sets
  difference = training_data(:,1:test_length) - repmat(test_data(i,:),[training_length 1]);
  distances = sum(difference.^2,2);
  % rank distacnes to find k nearest neighbours
  [~, indices] = sort(distances);
  class_counts = zeros(1, nClasses);
  for j=1:k
    class_counts = class_counts + training_labels(indices(j),:);
  end
  
  % choose the class by max count
  indices = find(class_counts == max(class_counts));
  if (length(indices) == 1)
    % if no tie, the label randomly from options
    labels(i,indices(1)) = 1;
  else
    % for a tie, pick a random option from those avaible
    labels(i,indices(randi(length(indices)))) = 1;
  end
end

end 
