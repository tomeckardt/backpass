pub mod gradvec;
use gradvec::*;

pub mod optim;
use optim::*;

pub mod layers;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let t1 = GradVec::from((0..10).map(|x| x as f32));
        let t2 = GradVec::from([1.0, 3.0, -2.0, 2.0, 0.0, -1.0, 0.0, -2.0, -1.0, 0.0]);
        let gd = gd!(lr=0.005, &t1, &t2);
        for i in 0..100 {
            let t3 = &t1 + &t2;
            let t4 = (&t3 * &t3).abs();
            gd.step(&t4);
            if i > 90 {
                println!("{}", t4.data.borrow().iter().copied().sum::<f32>());
            }
        }
    }
}
