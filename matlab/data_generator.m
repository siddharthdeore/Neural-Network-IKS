clc; clear ; close;
l1=[1 0]';
l2=[1 0]';
l3=[1 0]';
itr=1;
for ii=0:10:360
    q0=ii*pi/180;
    for jj=0:10:360
        q1=jj*pi/180;
        for kk=0:10:360
            q2=kk*pi/180+0*randn*0.2745;
            [EE,J1,J2]=FK(q0,q1,q2);
            %state(itr,:)=[EE',J1',J2'];
            %if(EE(2)>=0 && EE(1)>=0 && J1(2)>=0 && J1(1)>=0 && J2(2)>=0 && J2(1)>=0 )
            data(itr,:)=[EE' q0 q1 q2];
            itr=itr+1;
            %end
        end
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
plot(data(1:40000,1),data(1:40000,2),'k.');
hold on
plot(data(40000:47000,1),data(40000:47000,2),'b.');
plot(data(47000:end,1),data(47000:end,2),'r.');
%plot(train(:,1),train(:,2),'k.');
axis equal

%{ 
for i=1:length(state)
plot([0,state(i,3),state(i,5),state(i,1)],[0,state(i,4),state(i,6),state(i,2)])
hold on
plot(state(i,1),state(i,2),'rd')
axis equal
axis([-3 3 -3 3])
end
csvwrite('trains.csv',train);
csvwrite('tests.csv',test);
csvwrite('preds.csv',data);
csvwrite('trains.csv',data(1:40000,:));
csvwrite('tests.csv',data(40000:47000,:));
csvwrite('preds.csv',data(47000:end,:));
%}
csvwrite('trains.csv',data(1:40000,:));
csvwrite('tests.csv',data(40000:47000,:));
csvwrite('preds.csv',data(47000:end,:));

%%


