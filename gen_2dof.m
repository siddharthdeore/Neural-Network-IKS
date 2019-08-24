clc; clear ; close;
l1=[1 0]';
itr=1;
for ii=0:5:360
    q0=ii*pi/180+randn*0*pi/180;
    for jj=0:5:360
        q1=jj*pi/180+randn*0*pi/180;
            [EE,J1]=fk2dof(q0,q1);
            %state(itr,:)=[EE',J1',J2'];
            %if(EE(2)>=0 && EE(1)>=0 && J1(2)>=0 && J1(1)>=0 && J2(2)>=0 && J2(1)>=0 )
            data(itr,:)=[EE' q0 q1];
            itr=itr+1;
            %end
    end
end
%{
index=randperm(length(data),40000);
train=data(index,:);
data(index,:)=[];

index=randperm(length(data),7000);
test=data(index,:);
data(index,:)=[];
plot(train(:,1),train(1:40000,2),'k.');
hold on
plot(test(:,1),test(40000:47000,2),'b.');
plot(data(47000:end,1),data(47000:end,2),'r.');

%}
plot(data(:,1),data(:,2),'k.');
axis equal

%{ 
for i=1:length(state)
plot([0,state(i,3),state(i,5),state(i,1)],[0,state(i,4),state(i,6),state(i,2)])
hold on
plot(state(i,1),state(i,2),'rd')
axis equal
axis([-3 3 -3 3])
end
csvwrite('train_2dof.csv',data(1:40000,:));
csvwrite('test_2dof.csv',data(40000:47000,:));
csvwrite('pred_2dof.csv',data(47000:end,:));
%}
csvwrite('train_2dof.csv',data(1:5000,:));
csvwrite('test_2dof.csv',data(5000:5300,:));
csvwrite('pred_2dof.csv',data(5300:end,:));

function [ee,J1]=fk2dof(q0,q1)
l1=[1;0];
J1=rz(q0)*l1;
ee=J1+rz(q0+q1)*l1;
end

