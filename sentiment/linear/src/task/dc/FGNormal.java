package task.dc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import text.Dict;
import training.DataPoint;

public class FGNormal extends DCFeatureGenerator {
	//freq or exist
	boolean no_freq;
	public FGNormal(FeatureCounter g,boolean no_f){
		super(g);
		no_freq = no_f;
	}
	public FGNormal(FeatureCounter g){
		super(g);
		no_freq = true;
	}
	
	protected Object[] toData(List<HashMap<Integer,Integer>>[] l,Dict dict,boolean training){
		List<DataPoint>[] new_data = new ArrayList[l.length];
		for(int i=0;i<l.length;i++){
			new_data[i] = new ArrayList<DataPoint>();
			for(HashMap<Integer,Integer> point : l[i]){
				int [] new_index = new int[point.size()];
				int temp_count = 0;
				for(int x: point.keySet())
					new_index[temp_count++] = x;
				Arrays.sort(new_index);
				if(no_freq)
					new_data[i].add(new DataPoint(i,new_index));
				else{
					double [] new_v = new double[point.size()];
					for(int ii=0;ii<new_index.length;ii++)
						new_v[ii] = point.get(new_index[ii]);
					new_data[i].add(new DataPoint(i,new_index,new_v));
				}
			}
		}
		return new Object[]{new_data,dict};
	}
}
