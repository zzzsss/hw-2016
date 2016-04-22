package task.dc;

import text.*;
import training.DataPoint;
import java.util.*;

public abstract class DCFeatureGenerator {
	FeatureCounter counter;
	public DCFeatureGenerator(FeatureCounter g){
		counter = g;
	}
	
	public Object[] get_datapoints(List<Paragraph>[] pl,Dict d){
		boolean training = false;
		Dict dic = d;
		if(dic == null){
			dic = new Dict();
			training = true;
		}
		List<HashMap<Integer,Integer>>[] counts = counter.count(pl, dic);
		dic.close_dict();
		return(toData(counts,dic,training));
	}
	
	//return Object{List<DataPoint>[],Dict} --- if d==null then generate one
	abstract protected Object[] toData(List<HashMap<Integer,Integer>>[] l,Dict d,boolean training);
	
	
	/*	//change these
	//return Object{List<DataPoint>[],Dict} --- if d==null then generate one
	public Object[] get_datapoints(List<Paragraph>[] pl,Dict d){
		Dict dic = d;
		if(dic == null)
			dic = new Dict();
		List<DataPoint>[] ret = new List[pl.length];
		//get_feature
		for(int i=0;i<pl.length;i++){
			ret[i] = new ArrayList<DataPoint>();
			for(Paragraph p : pl[i])
				ret[i].add(get_one(p,dic,i));
		}
		dic.close_dict();
		return new Object[]{ret,dic};
	}
	
	
	*/
	
	
}
