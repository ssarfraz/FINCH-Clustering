 function [c,num_clust, mat]=get_merge(c,u,data)
  %% core procedure for mergeing in aLgorthm 1
    u_ =ind2vec(u); num_clust=size(u_,1);           
      
    if ~isempty(c)
     c=getC(c,u');
     else
        c=u';
    end   
 
   
    
    
    % can compute mean very fast uptill more than 5 million clusters.. otherwise resort to approx)
   
      if num_clust<=5e6
          [mat] =coolMean(data,c); 

      else 
      %%
       disp('resorting to approx combining method ...')
       % [just sanity ..probably will never run]..
       %just pick one vector from each group without doing any mean combining..for very very big data
         [~,ic,~] = unique(c,'last');
         mat= data(ic,:);
   
      end    
      
      
     function G=getC(G,u)  

    [~,~,ig]=unique(G);
    
    G=u(ig);
        
    end
         
             
         
  end
