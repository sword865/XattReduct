package Xreducer_fuzzy;

public class ITStyle_Lukasiewicz extends ImplicatorTnormStyle{
	public double getfuzzyTnromValue(double x, double y){
		return Math.max(x+y-1, 0);
	}
	public double getfuzzyImplicatorValue(double x, double y){
		return Math.min(1-x+y, 1);
	}
	public String getInformation(){
		String str="Implicator: Lukasiewicz\n" +
				"T-Norm: Lukasiewicz\n" +
				"Relation composition: Lukasiewicz\n";
		System.out.println(str);
		return str;
	}
	public String getInfor(){
		return "Lukasiewicz";
	}
}
