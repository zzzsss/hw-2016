package task.dc.sentiment;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.List;
import text.Paragraph;

public class Evaluator {
	String xml_tag;
	String goal_tag;
	public Evaluator(String xml,String goal){
		xml_tag = xml;
		goal_tag = goal;
	}
	public void eval(String origin_f,List<Double> res){
		int all=0;
		int right=0;
		try{
			//read xml style --- but don't check format
			FileInputStream in = new FileInputStream(origin_f);
			BufferedReader dr = new BufferedReader(new InputStreamReader(in,"UTF-8"));
			StringBuilder temp = new StringBuilder();
			String line="";
			while((line=dr.readLine()) != null){
				if(!line.isEmpty()){
					if(line.indexOf("<"+xml_tag) > -1){
						int l = line.indexOf(goal_tag);
						char temp_res = line.charAt(l+7);
						int temp_x = (int)(temp_res-'0');
						if(temp_x == res.get(all))
							right++;
						all++;
					}
					else if(line.indexOf("</"+xml_tag) > -1){
						temp = new StringBuilder();
					}
					else
						temp.append(line);
				}
			}
			dr.close();
			
			System.out.println(right+"/"+all+"--"+(double)right/all);
		}catch (Exception e){
            e.printStackTrace();
		}
	}
}
