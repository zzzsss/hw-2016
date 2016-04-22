package text;

import java.util.*;
import java.io.*;

public class Dict {
	boolean opened;
	int index;
	public HashMap<String,Integer> maps;	//not good design...
	
	private void init(){
		opened = true;
		index = 0;
		maps = new HashMap<String,Integer>();
	}
	public Dict(){
		init();
	}
	public Dict(File f){
		init();
		try{
			Scanner sc = new Scanner(f);
			index = sc.nextInt();
			while(sc.hasNext()){
				String x = sc.next();
				int t = sc.nextInt();
				maps.put(x, t);
			}
			sc.close();
			opened = false;
		}catch (FileNotFoundException e){
            e.printStackTrace();
		}
	}
	public void open_dict(){
		opened = true;
	}
	public void close_dict(){
		opened = false;
	}
	public int add(String x){
		if(x.isEmpty() || x.matches("[ \t]*"))
			return -1;
		if(opened){
			Integer i = maps.get(x);
			if(i==null){
				maps.put(x, index);
				index ++;
				return index-1;
			}
			else
				return i;
		}else
			return index(x);
	}
	public int index(String x){
		Integer i = maps.get(x);
		if(i==null){
			return -1;
		}
		else
			return i;
	}
	public int get_length(){
		return index;
	}
	
	public void write(String filename){
		try{
			FileOutputStream out = new FileOutputStream(filename);
			PrintStream p = new PrintStream(out);
			p.println(index);
			for(Map.Entry<String,Integer> x : maps.entrySet()){
				p.println(x.getKey()+" "+x.getValue());
			}
			p.close();
		}catch (FileNotFoundException e){
            e.printStackTrace();
        }
	}
}
