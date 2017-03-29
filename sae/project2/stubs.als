/*
 * Static model: Signatures
 *
 * The model should contain the following (and potentially other) signatures.
 * If necessary, you have to make some of the signatures abstract and
 * make them extend other signatures.
 */

// Open questions: One passenger is able to make several bookings flights, that depart at the same time
// seats and aircrafts composition relation bla how to model

sig Aircraft extends PassengerLocation {
	seats: some Seat,
	flights: set Flight,
	location: one AircraftLocation
}{
	all disj f1, f2: flights | isBefore[getDeparture[f1], getDeparture[f2]] => isBefore[getArrival[f1], getDeparture[f2]]
	no disj f1, f2: flights | getDeparture[f1] = getDeparture[f2]
	all f: flights | getDestination[f] = getOrigin[getNextFlight[f, flights]] or no getLaterFlights[f, flights]
}

sig Airline {
	aircrafts: set Aircraft,
	flight_routes: set Flight
}

sig Airport extends AircraftLocation { }

sig Booking {
	passengers: some Passenger,
	category: one Class,
	flights: some Flight
}{
	all disj f1, f2: flights | isBefore[getDeparture[f1], getDeparture[f2]] => isBefore[getArrival[f1], getDeparture[f2]]
	all disj f1, f2: flights | getDeparture[f1] != getDeparture[f2]
}

sig RoundTrip extends Booking { }{ getOrigin[getFirstFlight[this]] = getDestination[getLastFlight[this]] }

abstract sig Class { }
one sig First_Class extends Class {}
one sig Business_Class extends Class {}
one sig Economy_Class extends Class {}

sig Flight {
	operators: some Airline,
	aircraft: one Aircraft,
	bookings: set Booking,
	departure_time: one Time,
	arrival_time: one Time,
	departure_airport: one Airport,
	arrival_airport: one Airport
}{
	isBefore[departure_time, arrival_time] &&
	departure_airport != arrival_airport
}

fact {
	all f: Flight, b: Booking | f in getFlightsInBooking[b] <=> b in f.bookings // ensures that booking which uses this flight is scheduled on the flihgt and vice versa
	all f: Flight, o: Airline | f in o.flight_routes <=> o in f.operators // ensures that airline which operates a flight, has this flight in flight_routes and vice versa
	all f: Flight, a: Aircraft | f in getFlights[a] <=> a in f.aircraft // ensures that Aircraft which is used for the flight, has this flight in flights and vice versa
}

sig Passenger {
	bookings: set Booking,
	location: one PassengerLocation
}

abstract sig Seat { }
sig EconomySeat extends Seat {}
sig BusinessSeat extends EconomySeat {}
sig FirstClassSeat extends BusinessSeat{}
fact { all s: Seat | #{a: Aircraft | s in getSeats[a]} = 1 }

abstract sig Location {}
abstract sig PassengerLocation extends Location {}
abstract sig AircraftLocation extends Location {}
lone sig Unknown extends PassengerLocation {}
lone sig InAir extends AircraftLocation {}

sig Time { after: lone Time }{ isBefore[this, after]  && !isBefore[after, this] } // ensures no cycles exist in the timeline
fact { (one t: Time | Time = t.*after) } // ensures that all times have a common predecessor

pred show {#Unknown = 1}
run show for 4

pred static_instance_4 // why can't I constrain it to use only 1 Booking in the models it creates?
{
	some disj b1, b2: Booking | #b1.flights = 1 && #b2.flights = 1 && getFirstFlight[b1] = getFirstFlight[b2] && 
			getFirstFlight[b1].departure_airport != getFirstFlight[b2].departure_airport &&
			getFirstFlight[b1].arrival_airport != getFirstFlight[b2].arrival_airport
	all f1,f2:Flight | f1.aircraft = f2.aircraft
	
}
run static_instance_4 for 4 but exactly 2 Passenger, exactly 2 Seat, exactly 1 Airline

pred static_instance_5 // why can't I constrain it to use only 1 Booking in the models it creates?
{
	{all p: Passenger | #p.bookings.flights = 3}
	{all p: Passenger | getFirstFlight[p.bookings].aircraft = getLastFlight[p.bookings].aircraft and not getNextFlight[getFirstFlight[p.bookings], getFirstFlight[p.bookings]].aircraft = getFirstFlight[p.bookings].aircraft}
	
}
run static_instance_5 for 4 but exactly 1 Passenger, exactly 2 Seat, exactly 1 Airline, exactly 2 Aircraft
/*
 * Static model: Predicates
 */
// True iff t1 is strictly before t2.
pred isBefore[t1, t2: Time] { t2 in t1.^after } // Since time is totally ordered, if not in transitiv closure, first time is before second

/*
 * Static model: Functions
 */
// Returns the departure time of the given flight.
fun getDeparture[f: Flight]: Time { f.departure_time }
// Returns the arrival time of the given flight.
fun getArrival[f: Flight]: Time { f.arrival_time }
// Returns the airport the given flight departs from.
fun getOrigin[f: Flight]: Airport { f.departure_airport }
// Returns the destination airport of the given flight. 
fun getDestination[f: Flight]: Airport { f.arrival_airport }
// Returns the first flight of the given booking.
fun getFirstFlight[b: Booking]: Flight { // only flight in set that has an earlier departure than every other flight in set
	{f: b.flights | all fl: (b.flights - f) | isBefore[getDeparture[f], getDeparture[fl]]}
}
// Returns the last flight of the given booking.
fun getLastFlight[b: Booking]: Flight { // only flight in set that has an later departure than every other flight in set
	{f: b.flights | all fl: (b.flights - f) | isBefore[getDeparture[fl], getDeparture[f]]}
}
// Returns all seats of the given aircraft. 
fun getSeats[a: Aircraft]: set Seat { a.seats }
// Returns all flights for which is given aircraft is used.
fun getFlights[a: Aircraft]: set Flight { a.flights }
// Returns all bookings booked by the given passenger.
fun getBookings[p: Passenger]: set Booking { p.bookings }
// Returns all flights contained in the given booking.
fun getFlightsInBooking[b: Booking]: set Flight { b.flights }

fun getLaterFlights[first: Flight, fs: Flight]: set Flight {
	{later: fs | isBefore[getDeparture[first], getDeparture[later]]}
}
fun getNextFlight[first: Flight, fs: Flight]: Flight{
	{n: getLaterFlights[first, fs] | all rest: (getLaterFlights[first, fs] - n) | isBefore[getDeparture[n], getDeparture[rest]]}
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
