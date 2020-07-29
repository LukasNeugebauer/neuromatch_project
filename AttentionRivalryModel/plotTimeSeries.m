function plot_TimeSeries(p)

colmat  = [0 .5 0;0 0 1];

cpsFigure(.8,1.6);
set(gcf,'Name',sprintf('%s Input: %1.2f %1.2f', p.condnames{p.cond}, p.input(1), p.input(2)));
subplot(5,1,1)
temp1 = p.r{3}(1,:);
temp2 = p.r{3}(2,:);
plot(p.tlist/1000, temp1,'Color',colmat(1,:));hold on;
plot(p.tlist/1000, temp2,'Color',colmat(2,:));
legend('Orientation A','Orientation B','Location','NorthEast');
ylim([0 1]);
xlim([0 max(p.tlist/1000)]);
ylabel('Response','FontSize',12)
title(sprintf('%s Input:%2.2f  %2.2f \n Binocular-summation Layer',p.condnames{p.cond},p.input(1),p.input(2)),...
    'FontSize',14);
set(gca,'FontSize',12,'box','off')

%Left eye
subplot(5,1,2);hold on
title(sprintf('Left-eye monocular layer'))
temp1 = p.r{1}(1,:);
temp2 = p.r{1}(2,:);
plot(p.tlist/1000, temp1,'Color',colmat(1,:))
plot(p.tlist/1000, temp2,'Color',colmat(2,:))
ylim([0 1])
ylabel('Response','FontSize',12)
set(gca,'FontSize',12,'box','off')

%Right eye
subplot(5,1,3);hold on
title(sprintf('Right-eye monocular layer'))
temp1 = p.r{2}(1,:);
temp2 = p.r{2}(2,:);
plot(p.tlist/1000, temp1,'Color',colmat(1,:))
plot(p.tlist/1000, temp2,'Color',colmat(2,:))
ylabel('Response','FontSize',12)
ylim([0 1])
set(gca,'FontSize',12,'box','off')

%Attention
subplot(5,1,4);hold on
title(sprintf('Attentional gain'))
temp1 = max(squeeze(p.r{6}(1,:))+1,0);
temp2 = max(squeeze(p.r{6}(2,:))+1,0);
plot(p.tlist/1000, temp1,'Color',colmat(1,:))
plot(p.tlist/1000, temp2,'Color',colmat(2,:))
xlim([0 max(p.tlist/1000)])
ylabel('Attentional gain','FontSize',12)
set(gca,'FontSize',12,'box','off')

%Inhibition from opponency layers
subplot(5,1,5);hold on
title(sprintf('Inhibition from opponency layers'))
temp1 = p.o{1}(1,:);
temp2 = p.o{2}(1,:);
plot(p.tlist/1000, temp1,'k-')
plot(p.tlist/1000, temp2,'k--')
xlim([0 max(p.tlist/1000)])
ylim([0 0.8])
xlabel('Time (sec)','FontSize',12)
ylabel('Mutual inhibition','FontSize',12)
set(gca,'FontSize',12,'box','off')
legend('suppression to LE','suppression to RE');
%tightfig;
drawnow;

    function h = cpsFigure(width,height,num,name)
        
        if exist('num','var') && ~isempty(num)
            h = figure(num);
        else
            h = figure;
        end
        if exist('name','var')
            set(gcf,'Name',name);
        end
        
        Position = get(h,'Position');
        Position(3) = width*Position(3);
        Position(4) = height*Position(4);
        set(h,'Position', Position,'color','w',...
            'PaperUnits','inches','PaperPosition',[0 0 width height]*4);
    end

end