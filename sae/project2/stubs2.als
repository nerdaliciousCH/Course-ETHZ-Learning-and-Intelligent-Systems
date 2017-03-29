/*
 * Static model: Signatures
 *
 * The model should contain the following (and potentially other) signatures.
 * If necessary, you have to make some of the signatures abstract and
 * make them extend other signatures.
 */

sig Aircraft { }

sig Airline { }

sig Airport { }

sig Booking { }

sig Class { }

sig Flight { }

sig Passenger { }

sig RoundTrip { }

sig Seat { }

sig Time { after: lone Time }{
	isBefore[this, after]  && !isBefore[after, this]
}
fact { one t: Time | Time = t.*after }

pred show {}
run show for 6
/*
 * Static model: Predicates
 */

// True iff t1 is strictly before t2.
pred isBefore[t1, t2: Time] { t2 in t1.^after }
