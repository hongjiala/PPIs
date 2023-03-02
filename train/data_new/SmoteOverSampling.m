 function    [sample,sampleLabel]=SmoteOverSampling(data,Label,ClassType,C,AttVector,k,type)
% Implement Smote algorithm.
% Like over-sampling method, it changes the training data distribution 
% by duplicating higher-cost training examples until the appearances 
% of different training examples are proportional to their costs. Here, 
% the duplication method is SMOTE.
%
%Usage;
%  [sample,sampleLabel]=SmoteOverSampling(data,Label,ClassType,C,AttVector,k,type)
%
%  sample: new training set after Smote-over-sampling to build cost-sensitive NN
%               format - row indexes attributes and column indexes
%               instances                             
%  sampleLabel: class labels for instances in new training set. 
%                       format - row vector
%  data: original training set.
%           format - row indexes attributes and column indexes instances
%  Label: class labels for instances in original training set
%            format - row vector
%  ClassType: class type
%  C: cost vector. C[i] is the cost of misclassifying the i-th class
%       instance, without considering the concrete class the instance has
%       been wrongly assigned to.
%  AttVector: attribute vector,1 presents for the corresponding attribute
%                   is nominal and 0 for numeric.
%  k: 'k' used in the FUNCTION SMOTE. see SMOTE for more information
%  type: 'type' used in the FUNCTION SMOTE. see SMOTE for more information
 




%check parameters
NumClass=size(ClassType,2);
if(size(C)~=NumClass)
    error('class number and cost matrix do not consistent.')
end
if(size(data,2)~=size(Label))
    error('instance numbers in data and Label do not consistent.')
end
if(size(data,1)~=length(AttVector))
    error('attribute numbers in data and AttVector do not consistent.')
end
%compute class distribution
ClassD=zeros(1,NumClass);
for i=1:NumClass
    id=find(Label==ClassType(i));
    ClassData{i}=data(:,id);
    ClassD(i)=length(id);
end
%compute new class distribution
cn=C./ClassD;
[tmp,baseClassID]=min(cn);
newClassD=floor(C/C(baseClassID)*ClassD(baseClassID));

%over sampling using SMOTE
sample=data;
sampleLabel=Label;
clear data Label;
i=1;
while(i<NumClass | i==NumClass )    
    if(newClassD(i)>ClassD(i))
        diff=newClassD(i)-ClassD(i);       
        s=SMOTE(ClassData{1},diff,k,type,AttVector);    
        sample=[sample s];
        sampleLabel=[sampleLabel repmat(ClassType(i),1,diff)];         
    end
    if(length(ClassData)>1)
        ClassData=ClassData(2:end);% delete after used
    end
    i=i+1;
end


%end

