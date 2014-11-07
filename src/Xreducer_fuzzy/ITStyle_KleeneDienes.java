package Xreducer_fuzzy;

public class ITStyle_KleeneDienes extends ImplicatorTnormStyle{
	public double getfuzzyTnromValue(double x, double y){
		return Math.min(x, y);
	}
	public double getfuzzyImplicatorValue(double x, double y){
		return Math.max(1-x, y);
	}
	public String getInfor(){
		return "KleeneDienes";
	}
	@Override
	public String getInformation() {
		// TODO Auto-generated method stub
		return "KleeneDienes";
	}
}
