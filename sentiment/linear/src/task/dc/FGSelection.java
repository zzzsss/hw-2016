package task.dc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import text.Dict;
import training.DataPoint;

public class FGSelection extends DCFeatureGenerator{
	int low_end,high_end;
	public FGSelection(FeatureCounter g,int l,int h){
		super(g);
		low_end = l;
		high_end = h;
	}
	
	protected Object[] toData(List<HashMap<Integer,Integer>>[] l,Dict dict,boolean training){
		if(! training){
			//act like a static method
			return (new FGNormal(counter,true)).toData(l, dict, training);
		}
		//based on existing, not frequency
		System.err.println("Original featre size is "+dict.get_length());
		String [] strings = new String[dict.get_length()];
		for(String s : dict.maps.keySet()){
			strings[dict.index(s)] = s;
		}
		//feature counting
		int counting[][] = new int[dict.get_length()][l.length];
		int deleting[] = new int[dict.get_length()];
		for(int i=0;i<l.length;i++){
			for(HashMap<Integer,Integer> one : l[i]){
				for(int x : one.keySet())
					counting[x][i] += 1;
			}
		}
		//choosing them based on counting
		Dict new_dict = new Dict();
		for(int i=0;i<dict.get_length();i++){
			boolean to_delete = true;
			for(int j=0;j<l.length;j++){
				if(counting[i][j]>=low_end && counting[i][j]<=high_end)
					to_delete = false;
			}
			if(to_delete)
				deleting[i] = 1;
			else
				new_dict.add(strings[i]);
		}
		new_dict.close_dict();
		System.err.println("Shrinking featre size to "+new_dict.get_length());
		//construct data
		List<DataPoint>[] new_data = new ArrayList[l.length];
		for(int i=0;i<l.length;i++){
			new_data[i] = new ArrayList<DataPoint>();
			for(HashMap<Integer,Integer> point : l[i]){
				int temp_count = 0;
				for(int onef : point.keySet())
					if(deleting[onef]==0)
						temp_count ++;
				int [] new_index = new int[temp_count];
				int counting_new_index_temp = 0;
				for(int origin_ind : point.keySet()){
					if(deleting[origin_ind]==0){
						String the_str = strings[origin_ind];
						int new_ind = new_dict.index(the_str);
						new_index[counting_new_index_temp] = new_ind;
						counting_new_index_temp++;
					}
				}
				Arrays.sort(new_index);
				new_data[i].add(new DataPoint(i,new_index));
			}
		}
		return new Object[]{new_data,new_dict};
	}
}
