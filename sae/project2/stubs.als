/*
 * Static model: Signatures
 *
 * The model should contain the following (and potentially other) signatures.
 * If necessary, you have to make some of the signatures abstract and
 * make them extend other signatures.
 */

open util/boolean

// possible that airport code is equal to booking ID
sig string {}

sig Aircraft {
	seats: some Seats
}

sig Airline {
	name: one string,
	aircrafts: set Aircraft
}

sig Airport {
	code: one string
}

fact unique_airport_code {
	all A1, A2: Airport | A1 != A2 => A1.code != A2.code
}

sig Booking {
	ID: one string,
	passengers: some Passenger,
	category: one Class,
	flights: some Flight
}{
	(no f1, f2: flights | isBefore[getDeparture[f1], getDeparture[f2]] && isBefore[getArrival[f2], getArrival[f1]])
	//&& (no f1, f2: flights | getDeparture[f1] = getDeparture[f2]) // for some reason not equivalent to line XX)1
}
fact ordered_flights_booking {
	(no b: Booking | no f1, f2: b.flights | getDeparture[f1] = getDeparture[f2]) && // XX)1
	(no b: Booking | no f1, f2: b.flights | getArrival[f1] = getDeparture[f2])
}

fact unique_booking_id{
	all B1, B2: Booking | B1 != B2 => B1.ID != B2.ID
}

sig RoundTrip extends Booking { }{
	getOrigin[getFirstFlight[this]] = getDestination[getLastFlight[this]]
}

abstract sig Class { }
one sig First_Class extends Class {}
one sig Business_Class extends Class {}
one sig Economy_Class extends Class {}

sig Flight {
	number: one string,
	bookings: set Booking,
	departure_time: one Time,
	arrival_time: one Time,
	departure_airport: one Airport,
	arrival_airport: one Airport
}{
	departure_time not in arrival_time.*after &&
	departure_airport != arrival_airport
}

fact {
	all f: Flight, b: Booking | f in b.flights <=> b in f.bookings
}

sig Passenger {
	bookings: set Booking
}

abstract sig Seat { }
sig EconomySeat extends Seat {}
sig BusinessSeat extends EconomySeat {}
sig FirstClassSeat extends BusinessSeat{}

sig Time {
	after: lone Time // lone, because we have a end in the timeline
}
fact {
	// The next two lines together ensure, that the time is a totally ordered list
	(one t: Time | Time = t.*after) && // ensures that all times have a common predecessor
	all t1, t2: Time | isBefore[t1, t2] => not isBefore[t2, t1] // ensures no cicles exist in the timeline
}

pred show{one b:Booking | #(b.flights) > 1}
run show

/*
 * Static model: Predicates
 */

// True iff t1 is strictly before t2.
pred isBefore[t1, t2: Time] {
	t2 in t1.^after // Since time is totally ordered, if not in transitiv closure, first time is before second
}

/*
 * Static model: Functions
 */

// Returns the departure time of the given flight.
fun getDeparture[f: Flight]: Time {
	f.departure_time
}

// Returns the arrival time of the given flight.
fun getArrival[f: Flight]: Time {
	f.arrival_time
}

// Returns the airport the given flight departs from.
fun getOrigin[f: Flight]: Airport {
	f.departure_airport
}

// Returns the destination airport of the given flight. 
fun getDestination[f: Flight]: Airport {
	f.arrival_airport
}

// Returns the first flight of the given booking.
fun getFirstFlight[b: Booking]: Flight {
	// only flight in set that has an earlier departure than every other flight in set
	{f: b.flights | all fl: (b.flights - f) | isBefore[getDeparture[f], getDeparture[fl]]}
}

// Returns the last flight of the given booking.
fun getLastFlight[b: Booking]: Flight {
	// only flight in set that has an later departure than every other flight in set
	{f: b.flights | all fl: (b.flights - f) | isBefore[getDeparture[fl], getDeparture[f]]}
}

// Returns all seats of the given aircraft. 
//fun getSeats[a: Aircraft]: set Seat {
	
//}

// Returns all flights for which is given aircraft is used.
//fun getFlights[a: Aircraft]: set Flight {}

// Returns all bookings booked by the given passenger.
fun getBookings[p: Passenger]: set Booking {
	p.bookings
}

// Returns all flights contained in the given booking.
fun getFlightsInBooking[b: Booking]: set Flight {
	b.flights
}

/*
 * Dynamic model: Functions
 */
/*
// Returns the state which comes after the given state.
fun getNextState[s: State]: State { ... } 

// Returns the location of the given passenger at the given time. 
fun getPassengerLocation[t: Time, p: Passenger]: PassengerLocation { ... }

// Returns the location of the given aircraft at the given time.
fun getAircraftLocation[t: Time, ac: Aircraft]: AircraftLocation { ... }

// Returns the time whose state the given State represents.
fun getTime[s: State]: Time { ... }
*/
